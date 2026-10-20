import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

class _SoundItem {
  const _SoundItem({
    required this.title,
    required this.source,
    required this.isAsset,
    this.filePath,
  });

  final String title;
  final String source;
  final bool isAsset;
  final String? filePath;

  Source toPlayerSource() =>
      isAsset ? AssetSource(source) : DeviceFileSource(source);
}

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
  static const int _defaultRequiredCrowDetections = 2;
  static const int _minRequiredCrowDetections = 1;
  static const int _maxRequiredCrowDetections = 5;

  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  final OnnxRuntime _onnx = OnnxRuntime();

  OrtSession? _session;
  StreamSubscription<Uint8List>? _audioSubscription;
  StreamSubscription<void>? _completeSubscription;

  final List<int> _pendingAudioBytes = [];
  final List<File> _importedSoundFiles = [];
  Directory? _importedSoundsDirectory;
  Timer? _playbackTimer;
  List<String> _labels = [];

  double _crowThreshold = 0.35;

  bool isRunning = false;
  bool isPlaying = false;
  bool _isAnalyzing = false;
  bool _modelReady = false;
  bool _resumeListeningAfterPlayback = false;
  bool _closing = false;

  int _consecutiveCrowDetections = 0;
  int _requiredCrowDetections = _defaultRequiredCrowDetections;

  // جلوگیری از پخش مجدد تا زمانی که تشخیص کلاغ قطع شود.
  bool _crowTriggerLatched = false;

  String status = 'برنامه متوقف است';
  String detectionText = 'هنوز تحلیلی انجام نشده است';

  @override
  void initState() {
    super.initState();
    unawaited(_loadImportedSounds());
  }

  Future<void> _loadImportedSounds() async {
    try {
      final appDirectory = await getApplicationDocumentsDirectory();
      final directory = Directory('${appDirectory.path}/imported_sounds');
      if (!await directory.exists()) {
        await directory.create(recursive: true);
      }
      final files = directory
          .listSync()
          .whereType<File>()
          .where((file) {
            final name = file.path.toLowerCase();
            return name.endsWith('.wav') ||
                name.endsWith('.mp3') ||
                name.endsWith('.m4a') ||
                name.endsWith('.aac') ||
                name.endsWith('.ogg') ||
                name.endsWith('.flac') ||
                name.endsWith('.opus');
          })
          .toList()
        ..sort((a, b) => a.path.toLowerCase().compareTo(b.path.toLowerCase()));
      _importedSoundsDirectory = directory;
      if (mounted) {
        setState(() {
          _importedSoundFiles
            ..clear()
            ..addAll(files);
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => status = 'خطا در بارگذاری فایل‌های صوتی: $e');
      }
    }
  }

  String _fileName(File file) {
    return file.path.split(Platform.pathSeparator).last;
  }

  Future<void> importAudioFiles({VoidCallback? onUpdated}) async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.audio,
        allowMultiple: true,
        withData: false,
      );
      if (result == null || result.files.isEmpty) return;

      final appDirectory = await getApplicationDocumentsDirectory();
      final directory = _importedSoundsDirectory ??
          Directory('${appDirectory.path}/imported_sounds');
      if (!await directory.exists()) {
        await directory.create(recursive: true);
      }
      var copied = 0;
      for (final picked in result.files) {
        final path = picked.path;
        if (path == null) continue;
        final lower = path.toLowerCase();
        final supported = ['.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.opus']
            .any(lower.endsWith);
        if (!supported) continue;

        final originalName = picked.name.replaceAll(
          RegExp(r'[^a-zA-Z0-9._-]'),
          '_',
        );
        final destination = File(
          '${directory.path}/${DateTime.now().microsecondsSinceEpoch}_$originalName',
        );
        await File(path).copy(destination.path);
        copied++;
      }
      await _loadImportedSounds();
      onUpdated?.call();
      if (mounted) {
        setState(() {
          status = copied > 0
              ? '$copied فایل صوتی وارد شد'
              : 'فایل صوتی پشتیبانی‌شده انتخاب نشد';
        });
      }
    } catch (e) {
      if (mounted) setState(() => status = 'خطا در واردکردن صدا: $e');
    }
  }

  List<_SoundItem> _availableSounds() {
    final sounds = <_SoundItem>[
      for (var number = 1; number <= 30; number++)
        _SoundItem(
          title: 'صدای پیش‌فرض $number',
          source: 'sounds/$number.wav',
          isAsset: true,
        ),
    ];
    for (final file in _importedSoundFiles) {
      sounds.add(
        _SoundItem(
          title: _fileName(file),
          source: file.path,
          filePath: file.path,
          isAsset: false,
        ),
      );
    }
    return sounds;
  }

  Future<void> deleteImportedSound(File file) async {
    try {
      if (await file.exists()) await file.delete();
      if (mounted) {
        setState(() {
          _importedSoundFiles.removeWhere(
            (item) => item.path == file.path,
          );
        });
      }
    } catch (e) {
      if (mounted) setState(() => status = 'خطا در حذف فایل: $e');
    }
  }

  Future<void> showAudioManager() async {
    await showDialog<void>(
      context: context,
      builder: (dialogContext) => StatefulBuilder(
        builder: (context, refreshDialog) => AlertDialog(
          backgroundColor: const Color(0xFF1C2A33),
          title: const Text('مدیریت فایل‌های صوتی'),
          content: SizedBox(
            width: double.maxFinite,
            height: 300,
            child: _importedSoundFiles.isEmpty
                ? const Center(
                    child: Text('هنوز صدایی از گوشی وارد نشده است.'),
                  )
                : ListView.builder(
                    shrinkWrap: true,
                    itemCount: _importedSoundFiles.length,
                    itemBuilder: (context, index) {
                      final file = _importedSoundFiles[index];
                      return ListTile(
                        dense: true,
                        title: Text(
                          _fileName(file),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        leading: const Icon(Icons.audiotrack),
                        trailing: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconButton(
                              tooltip: 'پیش‌شنیدن',
                              icon: const Icon(Icons.play_arrow),
                              onPressed: isPlaying
                                  ? null
                                  : () async {
                                      Navigator.pop(dialogContext);
                                      await _playSpecificSound(
                                        _SoundItem(
                                          title: _fileName(file),
                                          source: file.path,
                                          filePath: file.path,
                                          isAsset: false,
                                        ),
                                      );
                                    },
                            ),
                            IconButton(
                              tooltip: 'حذف',
                              icon: const Icon(Icons.delete_outline),
                              onPressed: () async {
                                await deleteImportedSound(file);
                                refreshDialog(() {});
                              },
                            ),
                          ],
                        ),
                      );
                    },
                  ),
          ),
          actions: [
            TextButton.icon(
              onPressed: () async {
                await importAudioFiles(
                  onUpdated: () => refreshDialog(() {}),
                );
              },
              icon: const Icon(Icons.add),
              label: const Text('افزودن از گوشی'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: const Text('بستن'),
            ),
          ],
        ),
      ),
    );
  }

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

      // فقط گونه‌هایی نمایش داده می‌شوند که امتیازشان
      // بیشتر از آستانه تشخیص باشد.
      final lines = <String>[];

      for (final index in indexes) {
        if (scores[index] <= _crowThreshold) continue;

        final confidence =
            (scores[index] * 100).clamp(0.0, 100.0);

        lines.add(
          '${_labels[index]} — ${confidence.toStringAsFixed(1)}٪',
        );
      }

      if (lines.isEmpty) {
        lines.add('گونه‌ای بالاتر از آستانه تشخیص نیست');
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
            ? 'کلاغ شناسایی شد ($_consecutiveCrowDetections/$_requiredCrowDetections)'
            : 'تحلیل انجام شد؛ کلاغ تأیید نشد';
      });

      if (crowDetected &&
          _consecutiveCrowDetections >= _requiredCrowDetections &&
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
    final sounds = _availableSounds();
    if (sounds.isEmpty) return;
    final selected = sounds[Random().nextInt(sounds.length)];
    await _playSpecificSound(selected);
  }

  Future<void> _playSpecificSound(_SoundItem sound) async {
    if (isPlaying || _closing) return;

    _resumeListeningAfterPlayback = isRunning;
    if (_resumeListeningAfterPlayback) {
      await stopListening();
    }

    try {
      await _completeSubscription?.cancel();
      _completeSubscription = null;
      _playbackTimer?.cancel();
      _playbackTimer = null;

      if (!mounted || _closing) return;

      setState(() {
        isPlaying = true;
        status = 'در حال پخش ${sound.title}';
      });

      _completeSubscription = _player.onPlayerComplete.listen((_) {
        unawaited(_finishPlayback('پخش صدا تمام شد'));
      });

      // شنونده مدت فایل پیش از شروع پخش ثبت می‌شود.
      final durationFuture = _player.onDurationChanged.first;
      await _player.play(sound.toPlayerSource());

      try {
        final duration = await durationFuture.timeout(
          const Duration(seconds: 4),
        );
        if (duration.inMilliseconds > 0 && duration.inSeconds >= 20) {
          // برای فایل‌های ۲۰ ثانیه و بیشتر، ۲۰ تا ۱۰۰ درصد انتخاب می‌شود.
          final percent = 20 + Random().nextInt(81);
          final playDuration = Duration(
            milliseconds: (duration.inMilliseconds * percent / 100).round(),
          );
          if (percent < 100) {
            _playbackTimer = Timer(playDuration, () {
              unawaited(stopPlayback());
            });
          }
        }
      } catch (_) {
        // اگر دستگاه مدت فایل را گزارش نکند، فایل تا پایان کامل پخش می‌شود.
      }
    } catch (e) {
      await _finishPlayback('خطا در پخش صدا: $e');
    }
  }

  Future<void> stopPlayback() async {
    if (!isPlaying) return;

    _playbackTimer?.cancel();
    _playbackTimer = null;
    _consecutiveCrowDetections = 0;
    _crowTriggerLatched = false;

    try {
      await _player.stop();
    } catch (_) {}

    await _finishPlayback('پخش صدا قطع شد');
  }

  Future<void> _finishPlayback(String message) async {
    if (!isPlaying) return;

    _playbackTimer?.cancel();
    _playbackTimer = null;
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
    _playbackTimer?.cancel();

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
    _playbackTimer?.cancel();
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

              // تنظیم آستانه تشخیص
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
                        textDirection: TextDirection.rtl,
                        children: [
                          const Expanded(
                            child: Text(
                              'آستانه اطمینان تشخیص',
                              textAlign: TextAlign.right,
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 14,
                              ),
                            ),
                          ),
                          Text(
                            '${(_crowThreshold * 100).round()}٪',
                            textAlign: TextAlign.left,
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
                      const SizedBox(height: 4),
                      Row(
                        textDirection: TextDirection.rtl,
                        children: [
                          const Expanded(
                            child: Text(
                              'تعداد تشخیص متوالی کلاغ',
                              textAlign: TextAlign.right,
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 13,
                              ),
                            ),
                          ),
                          DropdownButton<int>(
                            value: _requiredCrowDetections,
                            dropdownColor: const Color(0xFF1C2A33),
                            items: List.generate(
                              _maxRequiredCrowDetections,
                              (index) => DropdownMenuItem<int>(
                                value: index + 1,
                                child: Text('${index + 1} بار'),
                              ),
                            ),
                            onChanged: (value) {
                              if (value == null) return;
                              setState(() {
                                _requiredCrowDetections = value;
                                _consecutiveCrowDetections = 0;
                                _crowTriggerLatched = false;
                              });
                            },
                          ),
                        ],
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
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const Text(
                          'گونه‌های شناسایی‌شده',
                          textAlign: TextAlign.right,
                          textDirection: TextDirection.rtl,
                          style: TextStyle(
                            fontSize: 17,
                            fontWeight: FontWeight.bold,
                            color: Colors.greenAccent,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          detectionText,
                          textAlign: TextAlign.right,
                          textDirection: TextDirection.rtl,
                          style: const TextStyle(
                            fontSize: 16,
                            height: 1.7,
                          ),
                        ),
                      ],
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

              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: isPlaying ? null : playRandomSound,
                      icon: const Icon(Icons.shuffle),
                      label: const Text('پخش تصادفی'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: showAudioManager,
                      icon: const Icon(Icons.library_music),
                      label: const Text('مدیریت صداها'),
                    ),
                  ),
                ],
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
