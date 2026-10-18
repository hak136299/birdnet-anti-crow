
import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

void main() {
  runApp(const CrowApp());
}

class CrowApp extends StatelessWidget {
  const CrowApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'دفع کلاغ',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const CrowHomePage(),
    );
  }
}

class CrowHomePage extends StatefulWidget {
  const CrowHomePage({super.key});

  @override
  State<CrowHomePage> createState() => _CrowHomePageState();
}

class _CrowHomePageState extends State<CrowHomePage> {
  static const String _modelAsset =
      'assets/models/BirdNET+_V3.0-preview3.1_Global_10K-pruned_FP16.onnx';

  static const String _labelsAsset =
      'assets/models/BirdNET+_V3.0-preview3.1_Global_10K-pruned_Labels.csv';

  static const int _inputSampleRate = 16000;
  static const int _modelSampleRate = 32000;
  static const int _windowSeconds = 3;
  static const int _inputWindowSamples =
      _inputSampleRate * _windowSeconds;
  static const int _inputWindowBytes = _inputWindowSamples * 2;
  static const double _crowThreshold = 0.35;

  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  final OnnxRuntime _onnx = OnnxRuntime();

  OrtSession? _session;
  StreamSubscription<Uint8List>? _audioSubscription;
  StreamSubscription<void>? _completeSubscription;

  final List<int> _pendingAudioBytes = [];
  List<String> _labels = [];

  bool isRunning = false;
  bool isPlaying = false;
  bool _isAnalyzing = false;
  bool _modelReady = false;
  bool _resumeListeningAfterPlayback = false;
  bool _closing = false;

  int _consecutiveCrowDetections = 0;
  int _lastCrowAnalysisCount = 0;

  String status = 'برنامه متوقف است';
  String detectionText = 'هنوز تحلیلی انجام نشده است';

  Future<void> _initializeModel() async {
    if (_modelReady) return;

    if (mounted) {
      setState(() {
        status = 'در حال بارگذاری مدل BirdNET...';
      });
    }

    final csv = await rootBundle.loadString(_labelsAsset);
    _labels = _parseLabels(csv);

    final session = await _onnx.createSessionFromAsset(_modelAsset);

    if (!session.inputNames.contains('input')) {
      await session.close();
      throw StateError('ورودی input در مدل پیدا نشد');
    }

    if (!session.outputNames.contains('predictions')) {
      await session.close();
      throw StateError('خروجی predictions در مدل پیدا نشد');
    }

    _session = session;
    _modelReady = true;
  }

  List<String> _parseLabels(String csv) {
    final lines = csv
        .split('\n')
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList();

    if (lines.length < 2) {
      throw const FormatException('فایل برچسب‌های BirdNET خالی است');
    }

    final header = lines.first.split(';');
    final indexColumn = header.indexOf('idx');
    final nameColumn = header.indexOf('com_name');

    if (indexColumn < 0 || nameColumn < 0) {
      throw const FormatException(
        'ستون‌های idx و com_name در فایل برچسب پیدا نشدند',
      );
    }

    final labels = <int, String>{};

    for (var i = 1; i < lines.length; i++) {
      final columns = lines[i].split(';');

      if (columns.length <= max(indexColumn, nameColumn)) continue;

      final index = int.tryParse(columns[indexColumn].trim());
      final name = columns[nameColumn].trim();

      if (index != null && name.isNotEmpty) {
        labels[index] = name;
      }
    }

    if (labels.isEmpty) {
      throw const FormatException('هیچ نام گونه‌ای بارگذاری نشد');
    }

    final maxIndex = labels.keys.reduce(max);
    return List<String>.generate(
      maxIndex + 1,
      (index) => labels[index] ?? 'گونه شماره $index',
    );
  }

  Future<void> startListening() async {
    if (isRunning || isPlaying || _closing) return;

    try {
      await _initializeModel();

      final hasPermission = await _recorder.hasPermission();

      if (!hasPermission) {
        if (!mounted) return;
        setState(() {
          status = 'مجوز میکروفون داده نشده است';
        });
        return;
      }

      _pendingAudioBytes.clear();
      _consecutiveCrowDetections = 0;
      _lastCrowAnalysisCount = 0;

      final audioStream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: _inputSampleRate,
          numChannels: 1,
        ),
      );

      if (!mounted || _closing) {
        await _recorder.stop();
        return;
      }

      setState(() {
        isRunning = true;
        status = 'شنود فعال است؛ در انتظار تحلیل صدا...';
        detectionText = 'در حال دریافت صدا';
      });

      _audioSubscription = audioStream.listen(
        (data) {
          if (!isRunning || isPlaying || _closing) return;

          _pendingAudioBytes.addAll(data);
          _processNextWindow();
        },
        onError: (error) {
          if (!mounted) return;
          setState(() {
            isRunning = false;
            status = 'خطا در دریافت صدای میکروفون';
          });
        },
      );
    } catch (e) {
      if (!mounted) return;

      setState(() {
        isRunning = false;
        status = 'خطا در شروع یا بارگذاری مدل: $e';
      });
    }
  }

  Future<void> stopListening() async {
    await _audioSubscription?.cancel();
    _audioSubscription = null;

    try {
      await _recorder.stop();
    } catch (_) {}

    _pendingAudioBytes.clear();

    if (!mounted) return;

    setState(() {
      isRunning = false;
      status = isPlaying ? 'در حال پخش صدا' : 'شنود متوقف است';
    });
  }

  void _processNextWindow() {
    if (_isAnalyzing ||
        !isRunning ||
        isPlaying ||
        _closing ||
        _pendingAudioBytes.length < _inputWindowBytes) {
      return;
    }

    final bytes = Uint8List.fromList(
      _pendingAudioBytes.sublist(0, _inputWindowBytes),
    );
    _pendingAudioBytes.removeRange(0, _inputWindowBytes);

    _isAnalyzing = true;

    unawaited(() async {
      try {
        if (isRunning && !isPlaying && !_closing) {
          await _analyzeAudio(bytes);
        }
      } catch (e) {
        if (mounted && isRunning && !isPlaying) {
          setState(() {
            status = 'خطا در تحلیل صدا: $e';
          });
        }
      } finally {
        _isAnalyzing = false;

        if (isRunning && !isPlaying && !_closing) {
          _processNextWindow();
        }
      }
    }());
  }

  Float32List _convertPcm16ToFloat32(Uint8List bytes) {
    final samples = Float32List(bytes.length ~/ 2);
    final data = ByteData.sublistView(bytes);

    for (var i = 0; i < samples.length; i++) {
      samples[i] = data.getInt16(i * 2, Endian.little) / 32768.0;
    }

    return samples;
  }

  Float32List _resample16kTo32k(Float32List input) {
    final output = Float32List(input.length * 2);

    for (var i = 0; i < input.length; i++) {
      output[i * 2] = input[i];

      output[i * 2 + 1] = i + 1 < input.length
          ? (input[i] + input[i + 1]) / 2.0
          : input[i];
    }

    return output;
  }

  Future<void> _analyzeAudio(Uint8List bytes) async {
    final session = _session;
    if (session == null || !_modelReady) return;

    final pcm = _convertPcm16ToFloat32(bytes);
    final audio = _resample16kTo32k(pcm);

    final inputTensor = await OrtValue.fromList(
      audio,
      [1, _modelSampleRate * _windowSeconds],
    );

    Map<String, OrtValue>? outputs;

    try {
      outputs = await session.run({'input': inputTensor});

      final predictionsTensor = outputs['predictions'];
      if (predictionsTensor == null) {
        throw StateError('خروجی predictions دریافت نشد');
      }

      final raw = await predictionsTensor.asFlattenedList();
      final scores = raw.map((value) => (value as num).toDouble()).toList();

      if (scores.isEmpty) {
        throw StateError('خروجی مدل خالی است');
      }

      final count = min(scores.length, _labels.length);
      final indexes = List<int>.generate(count, (index) => index);

      indexes.sort((a, b) => scores[b].compareTo(scores[a]));

      final topIndexes = indexes.take(5).toList();
      final lines = <String>[];

      for (final index in topIndexes) {
        final confidence = (scores[index] * 100).clamp(0.0, 100.0);
        lines.add('${_labels[index]} — ${confidence.toStringAsFixed(1)}٪');
      }

      // وجود نام Crow با امتیاز حداقل 0.35 یک تشخیص کلاغ محسوب می‌شود.
      var crowDetected = false;

      for (var i = 0; i < count; i++) {
        if (_labels[i].toLowerCase().contains('crow') &&
            scores[i] >= _crowThreshold) {
          crowDetected = true;
          break;
        }
      }

      if (crowDetected) {
        _consecutiveCrowDetections++;
      } else {
        _consecutiveCrowDetections = 0;
      }

      if (!mounted) return;

      setState(() {
        detectionText = lines.join('\n');
        status = crowDetected
            ? 'کلاغ شناسایی شد (${_consecutiveCrowDetections}/2)'
            : 'تحلیل انجام شد؛ کلاغ تأیید نشد';
      });

      // فقط یک بار برای هر جفت تشخیص متوالی صدا پخش می‌شود.
      if (crowDetected &&
          _consecutiveCrowDetections >= 2 &&
          _lastCrowAnalysisCount != _consecutiveCrowDetections &&
          isRunning &&
          !isPlaying) {
        _lastCrowAnalysisCount = _consecutiveCrowDetections;
        await playRandomSound();
      }
    } finally {
      await inputTensor.dispose();

      if (outputs != null) {
        for (final tensor in outputs.values) {
          await tensor.dispose();
        }
      }
    }
  }

  Future<void> playRandomSound() async {
    if (isPlaying || _closing) return;

    _resumeListeningAfterPlayback = isRunning;

    if (_resumeListeningAfterPlayback) {
      await stopListening();
    }

    try {
      final number = Random().nextInt(30) + 1;

      await _completeSubscription?.cancel();
      _completeSubscription = null;

      if (!mounted) return;

      setState(() {
        isPlaying = true;
        status = 'در حال پخش فایل شماره $number';
      });

      _completeSubscription = _player.onPlayerComplete.listen((_) {
        unawaited(_finishPlayback('پخش صدا تمام شد'));
      });

      await _player.play(AssetSource('sounds/$number.wav'));
    } catch (e) {
      await _finishPlayback('خطا در پخش صدا: $e');
    }
  }

  Future<void> stopPlayback() async {
    if (!isPlaying) return;

    try {
      await _player.stop();
    } catch (_) {}

    await _finishPlayback('پخش صدا قطع شد');
  }

  Future<void> _finishPlayback(String message) async {
    if (!isPlaying) return;

    await _completeSubscription?.cancel();
    _completeSubscription = null;

    final shouldResume = _resumeListeningAfterPlayback;
    _resumeListeningAfterPlayback = false;

    if (mounted) {
      setState(() {
        isPlaying = false;
        status = message;
      });
    }

    if (shouldResume && mounted && !_closing) {
      await startListening();
    }
  }

  Future<void> exitApp() async {
    if (isRunning) {
      await stopListening();
    }

    _closing = true;

    try {
      await _player.stop();
    } catch (_) {}

    await _completeSubscription?.cancel();

    if (!mounted) return;

    final shouldExit = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('خروج از برنامه'),
        content: const Text('آیا می‌خواهید از برنامه خارج شوید؟'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('انصراف'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('خروج'),
          ),
        ],
      ),
    );

    if (shouldExit == true) {
      await _session?.close();
      await _recorder.dispose();
      await _player.dispose();
      await SystemNavigator.pop();
    } else {
      _closing = false;
    }
  }

  @override
  void dispose() {
    _audioSubscription?.cancel();
    _completeSubscription?.cancel();
    _session?.close();
    _recorder.dispose();
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          'برنامهٔ دفع کلاغ',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              Icon(
                isPlaying
                    ? Icons.volume_up
                    : isRunning
                        ? Icons.hearing
                        : Icons.hearing_disabled,
                size: 64,
                color: isPlaying
                    ? Colors.orange
                    : isRunning
                        ? Colors.green
                        : Colors.grey,
              ),
              const SizedBox(height: 12),
              Text(
                status,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 16),
              ),
              const SizedBox(height: 12),
              Expanded(
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey.shade400),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: SingleChildScrollView(
                    child: Text(
                      'گونه‌های شناسایی‌شده:\n\n$detectionText',
                      style: const TextStyle(fontSize: 16),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: isRunning || isPlaying ? null : startListening,
                  child: const Text('شروع شنود'),
                ),
              ),
              const SizedBox(height: 8),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: isRunning ? stopListening : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    foregroundColor: Colors.white,
                  ),
                  child: const Text('توقف شنود'),
                ),
              ),
              const SizedBox(height: 8),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: isPlaying ? null : playRandomSound,
                  child: const Text('پخش تصادفی'),
                ),
              ),
              const SizedBox(height: 8),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: isPlaying ? stopPlayback : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.deepOrange,
                    foregroundColor: Colors.white,
                  ),
                  child: const Text('قطع پخش'),
                ),
              ),
              const SizedBox(height: 8),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: exitApp,
                  icon: const Icon(Icons.exit_to_app),
                  label: const Text('خروج از برنامه'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
