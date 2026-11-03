import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:record/record.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';
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
          seedColor: const Color(0xFF20C7B5),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF071923),
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
  static const MethodChannel _backgroundChannel = MethodChannel('anticrow/background');
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
  Timer? _playbackTimer;
  final List<File> _importedSounds = [];
  String? _currentImportedPath;

  final List<int> _pendingAudioBytes = [];
  List<String> _labels = [];

  double _crowThreshold = 0.35;

  bool isRunning = false;
  bool isPlaying = false;
  bool _isAnalyzing = false;
  bool _modelReady = false;
  bool _isStarting = false;
  bool _resumeListeningAfterPlayback = false;
  bool _closing = false;

  int _consecutiveCrowDetections = 0;
  int _requiredCrowDetections = 2;

  // جلوگیری از پخش مجدد تا زمانی که تشخیص کلاغ قطع شود.
  bool _crowTriggerLatched = false;

  String status = 'برنامه متوقف است';
  String detectionText = 'هنوز تحلیلی انجام نشده است';
  final List<String> _recentSpecies = [];

  @override
  void initState() {
    super.initState();
    unawaited(_loadImportedSounds());
  }

  Future<Directory> _soundDirectory() async {
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.path}${Platform.pathSeparator}imported_sounds');
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  Future<void> _loadImportedSounds() async {
    try {
      final dir = await _soundDirectory();
      final files = dir
          .listSync()
          .whereType<File>()
          .where((f) => _isAudioFile(f.path))
          .toList();
      files.sort((a, b) => a.path.toLowerCase().compareTo(b.path.toLowerCase()));
      if (mounted) setState(() {
        _importedSounds
          ..clear()
          ..addAll(files);
      });
    } catch (e) {
      if (mounted) setState(() => status = 'خطا در خواندن فایل‌های صوتی: $e');
    }
  }

  bool _isAudioFile(String path) => path.toLowerCase().endsWith('.wav');

  Future<void> _importAudioFiles() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['wav'],
        allowMultiple: false,
        withData: false,
      );
      if (result == null || result.files.isEmpty) return;

      final picked = result.files.first;
      final sourcePath = picked.path;
      if (sourcePath == null || sourcePath.isEmpty || !_isAudioFile(sourcePath)) {
        if (mounted) setState(() => status = 'فقط فایل WAV قابل افزودن است');
        return;
      }
      final source = File(sourcePath);
      if (!await source.exists()) return;

      final dir = await _soundDirectory();
      // فایل‌های واردشده با شماره 1 تا 30 ذخیره می‌شوند؛ شماره حذف‌شده آزاد می‌شود.
      final used = <int>{};
      for (final file in dir.listSync().whereType<File>()) {
        final match = RegExp(r'^(\d+)\.wav$', caseSensitive: false)
            .firstMatch(file.uri.pathSegments.last);
        final number = match == null ? null : int.tryParse(match.group(1)!);
        if (number != null && number >= 1 && number <= 30) used.add(number);
      }
      var slot = 1;
      while (used.contains(slot) && slot <= 30) { slot++; }
      if (slot > 30) {
        if (mounted) setState(() => status = 'هر ۳۰ شماره پر است؛ ابتدا یکی از فایل‌های شماره‌دار را حذف کنید');
        return;
      }
      await source.copy('${dir.path}${Platform.pathSeparator}$slot.wav');
      await _loadImportedSounds();
      if (mounted) setState(() => status = 'فایل جدید با نام $slot.wav ذخیره شد');
    } catch (e) {
      if (mounted) setState(() => status = 'خطا در افزودن فایل WAV: $e');
    }
  }

  Future<void> _deleteImportedSound(File file, StateSetter dialogSetState) async {
    if (_currentImportedPath == file.path && isPlaying) await stopPlayback();
    try {
      await file.delete();
      await _loadImportedSounds();
      dialogSetState(() {});
    } catch (e) {
      if (mounted) setState(() => status = 'خطا در حذف فایل: $e');
    }
  }

  Future<void> _previewImportedSound(File file) async {
    if (isPlaying) await stopPlayback();
    await _playSource(DeviceFileSource(file.path), label: file.uri.pathSegments.last, importedPath: file.path);
  }

  Future<void> _previewSoundNumber(int number, File? replacement) async {
    if (isPlaying) await stopPlayback();
    if (replacement != null) {
      await _playSource(
        DeviceFileSource(replacement.path),
        label: 'فایل شماره $number',
        importedPath: replacement.path,
      );
    } else {
      await _playSource(
        AssetSource('sounds/$number.wav'),
        label: 'فایل شماره $number',
      );
    }
  }

  BoxDecoration _greenPageGradient({double radius = 20}) => BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFF2E8B45),
            Color(0xFF176B36),
            Color(0xFF0B3D2A),
          ],
        ),
        borderRadius: BorderRadius.circular(radius),
        border: Border.all(color: const Color(0xFF8DE0A0).withOpacity(0.65)),
        boxShadow: const [
          BoxShadow(color: Color(0x5500180A), blurRadius: 14, offset: Offset(0, 6)),
        ],
      );

  Widget _gradientDialogButton({
    required String label,
    required VoidCallback? onPressed,
    Color color = const Color(0xFF18A7E8),
    IconData? icon,
  }) {
    final light = Color.lerp(color, Colors.white, 0.30)!;
    final dark = Color.lerp(color, Colors.black, 0.24)!;
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [light, color, dark],
        ),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFFB8F0FF), width: 1.2),
        boxShadow: [
          BoxShadow(color: dark.withOpacity(0.35), blurRadius: 7, offset: const Offset(0, 3)),
        ],
      ),
      child: TextButton(
        onPressed: onPressed,
        style: TextButton.styleFrom(
          foregroundColor: Colors.white,
          disabledForegroundColor: Colors.white70,
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 9),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null) ...[Icon(icon, size: 19), const SizedBox(width: 7)],
            Text(label, style: const TextStyle(fontWeight: FontWeight.w900)),
          ],
        ),
      ),
    );
  }

  Future<void> showAudioManager() async {
    await _loadImportedSounds();
    if (!mounted) return;
    await showDialog<void>(
      context: context,
      builder: (dialogContext) => Directionality(
        textDirection: TextDirection.rtl,
        child: StatefulBuilder(
          builder: (context, dialogSetState) => AlertDialog(
            backgroundColor: const Color(0xFF12472B),
            surfaceTintColor: Colors.transparent,
            title: const Text(
              'مدیریت فایل‌های صوتی',
              textAlign: TextAlign.center,
              textDirection: TextDirection.rtl,
            ),
            content: SizedBox(
              width: double.maxFinite,
              height: 390,
              child: Column(
                children: [
                  SizedBox(
                    width: double.infinity,
                    child: _gradientDialogButton(
                      label: 'افزودن فایل از گوشی',
                      icon: Icons.add_rounded,
                      color: const Color(0xFF0878F9),
                      onPressed: () async {
                        await _importAudioFiles();
                        await _loadImportedSounds();
                        dialogSetState(() {});
                      },
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'فقط فایل WAV پذیرفته می‌شود. برای جایگزینی، فایل شماره‌دار را حذف کنید و فایل جدید را اضافه کنید.',
                    textAlign: TextAlign.right,
                    textDirection: TextDirection.rtl,
                    style: TextStyle(color: Colors.greenAccent, fontSize: 12, height: 1.5),
                  ),
                  const SizedBox(height: 8),
                  const Align(
                    alignment: Alignment.centerRight,
                    child: Text(
                      'فایل‌های موجود (۱ تا ۳۰)',
                      textAlign: TextAlign.right,
                      textDirection: TextDirection.rtl,
                      style: TextStyle(fontWeight: FontWeight.w800, color: Colors.white),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Expanded(
                    child: ListView.builder(
                      itemCount: 30,
                      itemBuilder: (context, index) {
                        final number = index + 1;
                        File? replacement;
                        for (final file in _importedSounds) {
                          if (file.uri.pathSegments.last.toLowerCase() == '$number.wav') {
                            replacement = file;
                            break;
                          }
                        }
                        final fileName = '$number.wav';
                        return Container(
                          margin: const EdgeInsets.symmetric(vertical: 2),
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(begin: Alignment.topRight, end: Alignment.bottomLeft, colors: [Color(0xFF286F3C), Color(0xFF12432D)]),
                            borderRadius: BorderRadius.circular(10),
                            border: Border.all(color: const Color(0xFF75C98A).withOpacity(0.35)),
                          ),
                          child: ListTile(
                            dense: true,
                            title: Text(
                              'فایل شماره $number',
                              textDirection: TextDirection.rtl,
                              textAlign: TextAlign.right,
                              style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700),
                            ),
                            subtitle: Row(
                              textDirection: TextDirection.rtl,
                              children: [
                                Text(
                                  replacement == null ? 'صدای پیش‌فرض' : 'فایل جایگزین: ',
                                  textDirection: TextDirection.rtl,
                                  style: TextStyle(color: replacement == null ? Colors.white60 : Colors.greenAccent, fontSize: 11),
                                ),
                                if (replacement != null)
                                  Directionality(
                                    textDirection: TextDirection.ltr,
                                    child: Text(fileName, style: const TextStyle(color: Colors.greenAccent, fontSize: 11)),
                                  ),
                              ],
                            ),
                            leading: Container(
                              decoration: BoxDecoration(
                                gradient: const LinearGradient(colors: [Color(0xFF48DDBA), Color(0xFF087F68)]),
                                shape: BoxShape.circle,
                                border: Border.all(color: Colors.white70),
                              ),
                              child: IconButton(
                                tooltip: 'پخش فایل',
                                icon: const Icon(Icons.play_arrow_rounded, color: Colors.white),
                                onPressed: () async {
                                  Navigator.of(dialogContext).pop();
                                  await _previewSoundNumber(number, replacement);
                                },
                              ),
                            ),
                            trailing: replacement == null
                                ? const Icon(Icons.music_note_rounded, color: Colors.white38)
                                : Container(
                                    decoration: BoxDecoration(
                                      gradient: const LinearGradient(colors: [Color(0xFFFF7777), Color(0xFFB5122B)]),
                                      shape: BoxShape.circle,
                                      border: Border.all(color: Colors.white70),
                                    ),
                                    child: IconButton(
                                      tooltip: 'حذف فایل جایگزین',
                                      icon: const Icon(Icons.delete_outline_rounded, color: Colors.white),
                                      onPressed: () async {
                                        await _deleteImportedSound(replacement!, dialogSetState);
                                      },
                                    ),
                                  ),
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
            actions: [
              _gradientDialogButton(
                label: 'بستن',
                onPressed: () => Navigator.of(dialogContext).pop(),
                color: const Color(0xFF159BE8),
                icon: Icons.close_rounded,
              ),
            ],
          ),
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
    if (isRunning || _isStarting || isPlaying || _closing) return;

    _isStarting = true;
    if (mounted) {
      setState(() {
        status = 'در حال آماده‌سازی مدل و میکروفون...';
      });
    }

    try {
      // ابتدا مدل و برچسب‌ها آماده می‌شوند؛ خطای مدل جداگانه گزارش می‌شود.
      await _initializeModel();
      if (_closing || !mounted) return;

      final hasPermission = await _recorder.hasPermission();
      if (!hasPermission) {
        if (mounted) {
          setState(() {
            status = 'مجوز میکروفون داده نشده است';
            isRunning = false;
          });
        }
        return;
      }

      if (!preserveCrowLatch) {
        _consecutiveCrowDetections = 0;
        _crowTriggerLatched = false;
      }
      _pendingAudioBytes.clear();

      // کانال سرویس پس‌زمینه در بعضی خروجی‌ها ممکن است پیاده‌سازی نشده باشد.
      // نبودن آن نباید مانع شروع ضبط و شنود در حالت روشن بودن صفحه شود.
      try {
        await _backgroundChannel.invokeMethod<void>('start');
      } on MissingPluginException {
        // ضبط میکروفون مستقل از این کانال ادامه پیدا می‌کند.
      } on PlatformException {
        // خطای سرویس پس‌زمینه نباید شروع میکروفون را متوقف کند.
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
        try {
          await _backgroundChannel.invokeMethod<void>('stop');
        } catch (_) {}
        return;
      }

      _audioSubscription = audioStream.listen(
        (data) {
          if (!isRunning || isPlaying || _closing) return;
          _pendingAudioBytes.addAll(data);
          _processNextWindow();
        },
        onError: (Object error) {
          unawaited(_handleAudioStreamError(error));
        },
        cancelOnError: true,
      );

      setState(() {
        isRunning = true;
        status = 'شنود فعال است؛ در انتظار تحلیل صدا...';
        detectionText = 'در حال شنود و تحلیل صدا';
      });
    } catch (e) {
      // در صورت شکست هر مرحله، ضبط نیمه‌کاره و سرویس احتمالی پاک‌سازی شود.
      try {
        await _audioSubscription?.cancel();
      } catch (_) {}
      _audioSubscription = null;
      try {
        await _recorder.stop();
      } catch (_) {}
      try {
        await _backgroundChannel.invokeMethod<void>('stop');
      } catch (_) {}

      if (mounted) {
        setState(() {
          isRunning = false;
          status = 'خطا در شروع شنود: $e';
        });
      }
    } finally {
      _isStarting = false;
      if (mounted && !_closing) {
        setState(() {
          if (!isRunning && status == 'در حال آماده‌سازی مدل و میکروفون...') {
            status = 'شنود شروع نشد';
          }
        });
      }
    }
  }

  Future<void> _handleAudioStreamError(Object error) async {
    try {
      await _audioSubscription?.cancel();
    } catch (_) {}
    _audioSubscription = null;
    try {
      await _recorder.stop();
    } catch (_) {}
    try {
      await _backgroundChannel.invokeMethod<void>('stop');
    } catch (_) {}
    _pendingAudioBytes.clear();
    if (!mounted) return;
    setState(() {
      isRunning = false;
      _isAnalyzing = false;
      status = 'خطا در دریافت صدای میکروفون: $error';
    });
  }

  Future<void> stopListening() async {
    await _audioSubscription?.cancel();
    _audioSubscription = null;

    try {
      await _recorder.stop();
    } catch (_) {}

    try {
      await _backgroundChannel.invokeMethod<void>('stop');
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
    if (mounted) {
      setState(() => status = 'در حال تحلیل صدا...');
    }

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
        if (mounted) {
          setState(() {
            if (isRunning && !isPlaying && !_closing &&
                status == 'در حال تحلیل صدا...') {
              status = 'شنود فعال است؛ در انتظار تحلیل صدا...';
            }
          });
        }
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
      final detectedLines = <String>[];

      for (final index in indexes) {
        if (scores[index] <= _crowThreshold) continue;

        final confidence =
            (scores[index] * 100).clamp(0.0, 100.0);

        detectedLines.add(
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
        // گونه‌های تازه را به ابتدای فهرست منتقل می‌کنیم؛ هر گونه فقط یک‌بار
        // در فهرست می‌ماند و حداکثر پنج گونه آخر نمایش داده می‌شود.
        for (final line in detectedLines.reversed) {
          final speciesName = line.split(' — ').first;
          _recentSpecies.removeWhere(
            (item) => item.split(' — ').first == speciesName,
          );
          _recentSpecies.insert(0, line);
        }
        if (_recentSpecies.length > 5) {
          _recentSpecies.removeRange(5, _recentSpecies.length);
        }
        detectionText = _recentSpecies.isEmpty
            ? 'هنوز گونه‌ای بالاتر از آستانه شناسایی نشده است'
            : _recentSpecies.join('\n');
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
    if (isPlaying || _closing) return;

    final number = Random().nextInt(30) + 1;
    // اگر کاربر WAV شماره‌دار وارد کرده باشد، همان فایل جایگزین صدای پیش‌فرض آن شماره می‌شود.
    File? replacement;
    for (final file in _importedSounds) {
      if (file.uri.pathSegments.last.toLowerCase() == '$number.wav') {
        replacement = file;
        break;
      }
    }
    if (replacement != null) {
      await _playSource(
        DeviceFileSource(replacement.path),
        label: 'فایل شماره $number',
        importedPath: replacement.path,
      );
    } else {
      await _playSource(
        AssetSource('sounds/$number.wav'),
        label: 'فایل شماره $number',
      );
    }
  }

  Future<void> _playSource(
    Source source, {
    required String label,
    String? importedPath,
  }) async {
    if (isPlaying || _closing) return;

    _resumeListeningAfterPlayback = isRunning;
    if (_resumeListeningAfterPlayback) await stopListening();

    try {
      await _completeSubscription?.cancel();
      _completeSubscription = null;
      _playbackTimer?.cancel();
      _playbackTimer = null;

      if (!mounted || _closing) return;

      setState(() {
        isPlaying = true;
        _currentImportedPath = importedPath;
        status = 'در حال پخش $label';
      });

      _completeSubscription = _player.onPlayerComplete.listen((_) {
        unawaited(_finishPlayback('پخش صدا تمام شد'));
      });

      await _player.setSource(source);
      final duration = await _player.getDuration();
      if (!mounted || _closing || !isPlaying) return;
      await _player.resume();

      // فایل‌های ۲۰ ثانیه یا بیشتر: پخش تصادفی ۲۰ تا ۱۰۰ درصد.
      // فایل کوتاه‌تر از ۲۰ ثانیه: پخش کامل.
      if (duration != null && duration.inSeconds >= 20) {
        final percent = 20 + Random().nextInt(81);
        if (percent < 100) {
          final playDuration = Duration(
            milliseconds: duration.inMilliseconds * percent ~/ 100,
          );
          _playbackTimer = Timer(playDuration, () async {
            if (!isPlaying || _closing) return;
            try { await _player.stop(); } catch (_) {}
            await _finishPlayback('پخش صدا تمام شد');
          });
        }
      }
    } catch (e) {
      await _finishPlayback('خطا در پخش صدا: $e');
    }
  }

  Future<void> stopPlayback() async {
    if (!isPlaying) return;

    _consecutiveCrowDetections = 0;
    _crowTriggerLatched = false;
    _playbackTimer?.cancel();
    _playbackTimer = null;

    try {
      await _player.stop();
    } catch (_) {}

    await _finishPlayback('پخش صدا قطع شد');
  }

  Future<void> _finishPlayback(String message) async {
    _playbackTimer?.cancel();
    _playbackTimer = null;
    if (!isPlaying) return;

    _currentImportedPath = null;

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

  void showSettingsDialog() {
    showDialog<void>(
      context: context,
      builder: (dialogContext) => Directionality(
        textDirection: TextDirection.rtl,
        child: AlertDialog(
          backgroundColor: const Color(0xFF12472B),
          surfaceTintColor: Colors.transparent,
          title: const Text(
            'تنظیمات تشخیص',
            textAlign: TextAlign.center,
            textDirection: TextDirection.rtl,
            style: TextStyle(color: Color(0xFF42E2CB), fontWeight: FontWeight.w900),
          ),
          content: SingleChildScrollView(
            child: Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(gradient: const LinearGradient(begin: Alignment.topRight, end: Alignment.bottomLeft, colors: [Color(0xFF286F3C), Color(0xFF103D2A)]), borderRadius: BorderRadius.circular(16), border: Border.all(color: const Color(0xFF75C98A))),
                child: Column(
                  children: [
                    Row(
                      textDirection: TextDirection.rtl,
                      children: [
                        const Expanded(child: Text('آستانه اطمینان', textDirection: TextDirection.rtl, style: TextStyle(fontWeight: FontWeight.w700, fontSize: 13))),
                        Container(padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 5), decoration: BoxDecoration(color: const Color(0xFF16483F), borderRadius: BorderRadius.circular(10)), child: Text('${(_crowThreshold * 100).round()}٪', style: const TextStyle(color: Color(0xFF52E6C4), fontWeight: FontWeight.w900))),
                      ],
                    ),
                    SliderTheme(
                      data: SliderTheme.of(context).copyWith(
                        activeTrackColor: const Color(0xFF35D6B5),
                        inactiveTrackColor: const Color(0xFF31515A),
                        thumbColor: const Color(0xFF65F0D4),
                        overlayColor: const Color(0x2235D6B5),
                        trackHeight: 3,
                      ),
                      child: Slider(
                        value: _crowThreshold,
                        min: _minThreshold,
                        max: _maxThreshold,
                        divisions: ((_maxThreshold - _minThreshold) / _thresholdStep).round(),
                        label: '${(_crowThreshold * 100).round()}٪',
                        onChanged: (value) => setState(() {
                          _crowThreshold = double.parse(value.toStringAsFixed(2));
                          _consecutiveCrowDetections = 0;
                          _crowTriggerLatched = false;
                        }),
                      ),
                    ),
                    const Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [Text('10٪', style: TextStyle(color: Color(0xFF9DB5BD), fontSize: 11)), Text('پیش‌فرض: 35٪', style: TextStyle(color: Color(0xFF9DB5BD), fontSize: 11)), Text('95٪', style: TextStyle(color: Color(0xFF9DB5BD), fontSize: 11))]),
                    const SizedBox(height: 5),
                    Row(
                      textDirection: TextDirection.rtl,
                      children: [
                        const Expanded(child: Text('تعداد تشخیص متوالی کلاغ', textDirection: TextDirection.rtl, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700))),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          decoration: BoxDecoration(color: const Color(0xFF0A202B), borderRadius: BorderRadius.circular(12), border: Border.all(color: const Color(0xFF31515A))),
                          child: DropdownButtonHideUnderline(
                            child: DropdownButton<int>(
                              value: _requiredCrowDetections,
                              dropdownColor: const Color(0xFF12313D),
                              iconEnabledColor: const Color(0xFF42E2CB),
                              items: List.generate(5, (i) => i + 1).map((n) => DropdownMenuItem<int>(value: n, child: Text('$n بار'))).toList(),
                              onChanged: (value) {
                                if (value == null) return;
                                setState(() { _requiredCrowDetections = value; _consecutiveCrowDetections = 0; _crowTriggerLatched = false; });
                              },
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
          ),
          actions: [
            _gradientDialogButton(
              label: 'بستن',
              onPressed: () => Navigator.pop(dialogContext),
              color: const Color(0xFF159BE8),
              icon: Icons.close_rounded,
            ),
          ],
        ),
      ),
    );
  }

  void showAboutDialogBox() {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF15512F),
        surfaceTintColor: Colors.transparent,
        title: const Column(
          children: [
            Text(
              'درباره برنامه',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            SizedBox(height: 10),
            Text(
              'HAKnet',
              style: TextStyle(
                fontSize: 25,
                fontWeight: FontWeight.w900,
                letterSpacing: 2.5,
                color: Color(0xFFB9D9FF),
              ),
            ),
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
                color: Colors.white,
              ),
            ),
            SizedBox(height: 12),
            Text(
              'این برنامه با شنود صدای محیط و تحلیل صوتی، گونه‌های پرندگان را شناسایی می‌کند و امکان پخش تصادفی صداها برای دور کردن کلاغ‌ها را فراهم می‌سازد.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                height: 1.6,
                color: Color(0xFFE8EDF2),
              ),
            ),
            SizedBox(height: 14),
            Text(
              'نسخه 1.0.1',
              style: TextStyle(color: Colors.white),
            ),
            SizedBox(height: 8),
            Text(
              'سازنده: حسین عبدالهی خسروشاهی',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white),
            ),
          ],
        ),
        actions: [
          _gradientDialogButton(
            label: 'بستن',
            onPressed: () => Navigator.pop(context),
            color: const Color(0xFF159BE8),
            icon: Icons.close_rounded,
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
    _playbackTimer?.cancel();

    if (!mounted) return;

    final shouldExit = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('خروج از برنامه'),
        content: const Text('آیا می‌خواهید از برنامه خارج شوید؟'),
        actions: [
          _gradientDialogButton(
            label: 'انصراف',
            onPressed: () => Navigator.pop(context, false),
            color: const Color(0xFF159BE8),
            icon: Icons.close_rounded,
          ),
          _gradientDialogButton(
            label: 'خروج',
            onPressed: () => Navigator.pop(context, true),
            color: const Color(0xFFE53945),
            icon: Icons.exit_to_app_rounded,
          ),
        ],
      ),
    );

    if (shouldExit == true) {
      await _session?.close();
      await _recorder.dispose();
      await _player.dispose();
      const exitChannel = MethodChannel('anticrow/exit');
      await exitChannel.invokeMethod<void>('finishAndRemoveTask');
    } else {
      _closing = false;
    }
  }

  @override
  void dispose() {
    _audioSubscription?.cancel();
    _completeSubscription?.cancel();
    _playbackTimer?.cancel();
    _session?.close();
    _recorder.dispose();
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final thresholdSteps = ((_maxThreshold - _minThreshold) / _thresholdStep).round();
    final Color stateColor = isPlaying
        ? const Color(0xFFFFA447)
        : _isAnalyzing
            ? const Color(0xFF53A8FF)
            : isRunning
                ? const Color(0xFF35D6AE)
                : const Color(0xFF8A9AA5);
    final IconData stateIcon = _isStarting
        ? Icons.hourglass_top_rounded
        : isPlaying
        ? Icons.volume_up_rounded
        : _isAnalyzing
            ? Icons.graphic_eq_rounded
            : isRunning
                ? Icons.hearing_rounded
                : Icons.hearing_disabled_rounded;
    final String stateLabel = _isStarting
        ? 'در حال شروع شنود'
        : isPlaying
        ? 'در حال پخش صدا'
        : _isAnalyzing
            ? 'در حال تحلیل صدا'
            : isRunning
                ? 'شنود فعال است'
                : 'شنود متوقف است';

    Widget actionButton({
      required String label,
      required IconData icon,
      required Color color,
      required VoidCallback? onPressed,
      bool outlined = false,
      Color foregroundColor = const Color(0xFF061820),
    }) {
      final shape = RoundedRectangleBorder(borderRadius: BorderRadius.circular(16));
      final bool disabled = onPressed == null;
      final Color light = Color.lerp(color, Colors.white, 0.24)!;
      final Color dark = Color.lerp(color, Colors.black, 0.22)!;
      return SizedBox(
        height: 58,
        width: double.infinity,
        child: DecoratedBox(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [light, color, dark],
            ),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: light.withOpacity(0.85), width: 1.2),
            boxShadow: [
              BoxShadow(color: dark.withOpacity(0.35), blurRadius: 9, offset: const Offset(0, 4)),
              BoxShadow(color: light.withOpacity(0.22), blurRadius: 5, offset: const Offset(-1, -2)),
            ],
          ),
          child: ElevatedButton.icon(
            onPressed: onPressed,
            icon: Icon(icon, size: 21),
            label: Text(label, textAlign: TextAlign.center, style: TextStyle(fontWeight: FontWeight.w800, fontSize: 13, color: Colors.white)),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.transparent,
              foregroundColor: foregroundColor,
              disabledBackgroundColor: Colors.transparent,
              disabledForegroundColor: Colors.white54,
              shadowColor: Colors.transparent,
              elevation: 0,
              shape: shape,
              padding: const EdgeInsets.symmetric(horizontal: 8),
            ),
          ),
        ),
      );
    }

    Widget sectionTitle(String title, IconData icon) => Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        textDirection: TextDirection.rtl,
        children: [
          Icon(icon, color: const Color(0xFF35D6C0), size: 21),
          const SizedBox(width: 8),
          Text(title, textAlign: TextAlign.right, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w800, color: Colors.white)),
        ],
      ),
    );

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // تصویر مستقل از ارتفاع محتواست و همیشه تمام نمایشگر را می‌پوشاند.
          Image.asset(
            'assets/images/orchard_background.png',
            fit: BoxFit.cover,
            alignment: Alignment.topCenter,
          ),
          Container(color: const Color(0x5506251A)),
          SafeArea(
            child: LayoutBuilder(
              builder: (context, constraints) {
                // فضای رزرو شده برای عنوانی که داخل خود تصویر پس‌زمینه است.
                final double topReserve = (constraints.maxHeight * 0.17)
                    .clamp(120.0, 245.0)
                    .toDouble();
                return SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(12, 4, 12, 14),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(minHeight: constraints.maxHeight),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        SizedBox(height: topReserve),
                        AnimatedContainer(
                          duration: const Duration(milliseconds: 180),
                          width: double.infinity,
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 9),
                          decoration: BoxDecoration(
                            color: const Color(0xA61C663B),
                            border: Border.all(color: stateColor.withOpacity(0.85)),
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: Row(
                            textDirection: TextDirection.rtl,
                            children: [
                              Container(
                                width: 36,
                                height: 36,
                                decoration: BoxDecoration(
                                  color: stateColor.withOpacity(0.22),
                                  borderRadius: BorderRadius.circular(13),
                                ),
                                child: Icon(stateIcon, color: stateColor, size: 23),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.end,
                                  children: [
                                    Text(
                                      stateLabel,
                                      textDirection: TextDirection.rtl,
                                      textAlign: TextAlign.right,
                                      style: TextStyle(color: stateColor, fontSize: 15, fontWeight: FontWeight.w900),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      status,
                                      textDirection: TextDirection.rtl,
                                      textAlign: TextAlign.right,
                                      style: const TextStyle(color: Color(0xFFF0FFF3), fontSize: 12),
                                      maxLines: 2,
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(width: 8),
                              AnimatedContainer(
                                duration: const Duration(milliseconds: 180),
                                width: 52,
                                height: 52,
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    begin: Alignment.topLeft,
                                    end: Alignment.bottomRight,
                                    colors: isRunning
                                        ? const [Color(0xFFFF6262), Color(0xFFD71932), Color(0xFF9E1025)]
                                        : const [Color(0xFF7CF45B), Color(0xFF22C900), Color(0xFF087A16)],
                                  ),
                                  shape: BoxShape.circle,
                                  border: Border.all(color: Colors.white.withOpacity(0.9), width: 2),
                                  boxShadow: [
                                    BoxShadow(color: (isRunning ? Colors.red : Colors.green).withOpacity(0.35), blurRadius: 9, offset: const Offset(0, 3)),
                                  ],
                                ),
                                child: Material(
                                  color: Colors.transparent,
                                  shape: const CircleBorder(),
                                  child: InkWell(
                                    customBorder: const CircleBorder(),
                                    onTap: isPlaying || _isStarting
                                        ? null
                                        : (isRunning ? stopListening : startListening),
                                    child: Icon(
                                      isRunning ? Icons.stop_rounded : Icons.play_arrow_rounded,
                                      color: Colors.white,
                                      size: 34,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 12),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 3),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              sectionTitle('گونه‌های شناسایی‌شده بالاتر از آستانه', Icons.radar_rounded),
                              Container(
                                constraints: const BoxConstraints(minHeight: 180),
                                padding: const EdgeInsets.all(13),
                                decoration: BoxDecoration(
                                  color: const Color(0xA61C663B),
                                  borderRadius: BorderRadius.circular(14),
                                  border: Border.all(color: const Color(0xFF8DE0A0).withOpacity(0.75)),
                                ),
                                child: Text(
                                  detectionText,
                                  textAlign: TextAlign.right,
                                  textDirection: TextDirection.rtl,
                                  style: const TextStyle(fontSize: 14, height: 1.7, color: Color(0xFFF0FFF3)),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                        Row(children: [
                          Expanded(child: actionButton(label: 'پخش تصادفی', icon: Icons.shuffle_rounded, color: const Color(0xFF35D83A), onPressed: isPlaying ? null : playRandomSound)),
                          const SizedBox(width: 10),
                          Expanded(child: actionButton(label: 'قطع پخش', icon: Icons.stop_circle_rounded, color: const Color(0xFFB00020), onPressed: isPlaying ? stopPlayback : null)),
                        ]),
                        const SizedBox(height: 10),
                        Row(children: [
                          Expanded(child: actionButton(label: 'مدیریت فایل‌های صوتی', icon: Icons.library_music_rounded, color: const Color(0xFFFF7800), onPressed: showAudioManager, outlined: true)),
                          const SizedBox(width: 10),
                          Expanded(child: actionButton(label: 'تنظیمات', icon: Icons.tune_rounded, color: const Color(0xFF1976D2), foregroundColor: Colors.white, onPressed: showSettingsDialog)),
                        ]),
                        const SizedBox(height: 10),
                        Row(
                          children: [
                            Expanded(
                              child: actionButton(
                                label: 'خروج از برنامه',
                                icon: Icons.exit_to_app_rounded,
                                color: const Color(0xFF8E44AD),
                                foregroundColor: Colors.white,
                                onPressed: exitApp,
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: actionButton(
                                label: 'درباره برنامه',
                                icon: Icons.info_outline_rounded,
                                color: const Color(0xFF0878F9),
                                foregroundColor: Colors.white,
                                onPressed: showAboutDialogBox,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        const Center(
                          child: Text(
                            'HAKnet',
                            style: TextStyle(color: Color(0xFF8DE0A0), fontSize: 10, fontWeight: FontWeight.w800, letterSpacing: 1.2),
                          ),
                        ),
                        const SizedBox(height: 2),
                        const Center(
                          child: Text(
                            'نسخه 1.0.1',
                            style: TextStyle(color: Color(0xFFE0F4E5), fontSize: 11, letterSpacing: 0.5),
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
