
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
      title: 'ضد کلاغ حاج اصغر',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF00A878),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF101820),
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

  static const double _minThreshold = 0.10;
  static const double _maxThreshold = 0.95;
  static const double _thresholdStep = 0.05;

  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  final OnnxRuntime _onnx = OnnxRuntime();

  OrtSession? _session;
  StreamSubscription<Uint8List>? _audioSubscription;
  StreamSubscription<void>? _completeSubscription;

  final List<int> _pendingAudioBytes = [];
  List<String> _labels = [];

  double _crowThreshold = 0.35;

  bool isRunning = false;
  bool isPlaying = false;
  bool _isAnalyzing = false;
  bool _modelReady = false;
  bool _resumeListeningAfterPlayback = false;
  bool _closing = false;

  int _consecutiveCrowDetections = 0;

  // جلوگیری از پخش مجدد تا زمانی که تشخیص کلاغ قطع شود.
  bool _crowTriggerLatched = false;

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

  // preserveCrowLatch برای شروع شنود بعد از پخش خودکار است.
  Future<void> startListening({
    bool preserveCrowLatch = false,
  }) async {
    if (isRunning || isPlaying || _closing) return;

    try {
      await _initializeModel();

      if (_closing || !mounted) return;

      final hasPermission = await _recorder.hasPermission();

      if (!hasPermission) {
        if (!mounted) return;
        setState(() {
          status = 'مجوز میکروفون داده نشده است';
        });
        return;
      }

      _pendingAudioBytes.clear();

      if (!preserveCrowLatch) {
        _consecutiveCrowDetections = 0;
        _crowTriggerLatched = false;
      }

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
            status = 'خطا در دریافت صدای میکروفون: $error';
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

      final lines = <String>[];

      // نمایش پنج گونه با بالاترین امتیاز.
      for (final index in indexes.take(5)) {
        final confidence =
            (scores[index] * 100).clamp(0.0, 100.0);

        lines.add(
          '${_labels[index]} — ${confidence.toStringAsFixed(1)}٪',
        );
      }

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
        // زنجیره تشخیص قطع شد؛ اجازه تشخیص جدید داده می‌شود.
        _consecutiveCrowDetections = 0;
        _crowTriggerLatched = false;
      }

      if (!mounted) return;

      setState(() {
        detectionText = lines.join('\n');
        status = crowDetected
            ? 'کلاغ شناسایی شد ($_consecutiveCrowDetections/2)'
            : 'تحلیل انجام شد؛ کلاغ تأیید نشد';
      });

      if (crowDetected &&
          _consecutiveCrowDetections >= 2 &&
          !_crowTriggerLatched &&
          isRunning &&
          !isPlaying) {
        _crowTriggerLatched = true;
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

      if (!mounted || _closing) return;

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

    // پس از پخش خودکار، وضعیت قفل تشخیص حفظ می‌شود.
    // اگر شنود قبلاً فعال نبوده، دوباره شروع نمی‌شود.
    if (shouldResume && mounted && !_closing) {
      await startListening(preserveCrowLatch: true);
    }
  }

  void showAboutDialogBox() {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1C2A33),
        title: const Column(
          children: [
            Text(
              'HAKnet',
              style: TextStyle(
                fontSize: 30,
                fontWeight: FontWeight.w900,
                letterSpacing: 2.5,
                color: Colors.greenAccent,
              ),
            ),
            SizedBox(height: 12),
            Text('درباره برنامه'),
          ],
        ),
        content: const Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'ضد کلاغ حاج اصغر',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 12),
            Text('نسخه 1.0.0'),
            SizedBox(height: 8),
            Text(
              'سازنده: حسین عبدالهی خسروشاهی',
              textAlign: TextAlign.center,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('بستن'),
          ),
        ],
      ),
    );
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
    final thresholdSteps =
        ((_maxThreshold - _minThreshold) / _thresholdStep).round();

    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        backgroundColor: const Color(0xFF007F5F),
        foregroundColor: Colors.white,
        title: const Text(
          'ضد کلاغ حاج اصغر',
          style: TextStyle(
            fontSize: 23,
            fontWeight: FontWeight.w900,
            letterSpacing: 0.2,
          ),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              // نام HAKnet در گوشه بالای صفحه اصلی
              const Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'HAKnet',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w900,
                    letterSpacing: 2,
                    color: Colors.greenAccent,
                    shadows: [
                      Shadow(
                        color: Color(0x8000FF80),
                        blurRadius: 8,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 8),

              Icon(
                isPlaying
                    ? Icons.volume_up_rounded
                    : isRunning
                        ? Icons.hearing_rounded
                        : Icons.hearing_disabled_rounded,
                size: 54,
                color: isPlaying
                    ? Colors.orange
                    : isRunning
                        ? Colors.greenAccent
                        : Colors.grey,
              ),
              const SizedBox(height: 8),

              Text(
                status,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 15),
              ),
              const SizedBox(height: 8),

              // تنظیم آستانه تشخیص کلاغ
              Card(
                color: const Color(0xFF1C2A33),
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                  child: Column(
                    children: [
                      Row(
                        children: [
                          const Expanded(
                            child: Text(
                              'آستانه اطمینان تشخیص کلاغ',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 14,
                              ),
                            ),
                          ),
                          Text(
                            '${(_crowThreshold * 100).round()}٪',
                            style: const TextStyle(
                              fontWeight: FontWeight.w900,
                              color: Colors.greenAccent,
                              fontSize: 17,
                            ),
                          ),
                        ],
                      ),
                      Slider(
                        value: _crowThreshold,
                        min: _minThreshold,
                        max: _maxThreshold,
                        divisions: thresholdSteps,
                        label: '${(_crowThreshold * 100).round()}٪',
                        activeColor: Colors.greenAccent,
                        onChanged: (value) {
                          setState(() {
                            _crowThreshold =
                                double.parse(value.toStringAsFixed(2));

                            _consecutiveCrowDetections = 0;
                            _crowTriggerLatched = false;
                          });
                        },
                      ),
                      const Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text('10٪', style: TextStyle(fontSize: 12)),
                          Text(
                            'مقدار اولیه: 35٪',
                            style: TextStyle(fontSize: 12),
                          ),
                          Text('95٪', style: TextStyle(fontSize: 12)),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 8),

              Expanded(
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: const Color(0xFF1C2A33),
                    border: Border.all(color: Colors.white24),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: SingleChildScrollView(
                    child: Text(
                      'گونه‌های شناسایی‌شده:\n\n$detectionText',
                      textAlign: TextAlign.start,
                      style: const TextStyle(
                        fontSize: 16,
                        height: 1.7,
                      ),
                    ),
                  ),
                ),
              ),

              const SizedBox(height: 10),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: isRunning || isPlaying
                      ? null
                      : startListening,
                  icon: const Icon(Icons.hearing),
                  label: const Text('شروع شنود'),
                ),
              ),
              const SizedBox(height: 6),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: isRunning ? stopListening : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red.shade700,
                    foregroundColor: Colors.white,
                  ),
                  icon: const Icon(Icons.hearing_disabled),
                  label: const Text('توقف شنود'),
                ),
              ),
              const SizedBox(height: 6),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: isPlaying ? null : playRandomSound,
                  icon: const Icon(Icons.shuffle),
                  label: const Text('پخش تصادفی'),
                ),
              ),
              const SizedBox(height: 6),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: isPlaying ? stopPlayback : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.deepOrange,
                    foregroundColor: Colors.white,
                  ),
                  icon: const Icon(Icons.stop_circle),
                  label: const Text('قطع پخش'),
                ),
              ),
              const SizedBox(height: 6),

              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: showAboutDialogBox,
                  icon: const Icon(Icons.info_outline),
                  label: const Text('دربارهٔ برنامه'),
                ),
              ),
              const SizedBox(height: 6),

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

