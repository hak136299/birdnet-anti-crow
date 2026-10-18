
import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';

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
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();

  StreamSubscription<Uint8List>? _audioSubscription;
  StreamSubscription<void>? _completeSubscription;

  bool isRunning = false;
  bool isPlaying = false;
  bool _resumeListeningAfterPlayback = false;

  String status = 'برنامه متوقف است';

  int _receivedBytes = 0;

  Future<void> startListening() async {
    if (isRunning || isPlaying) return;

    try {
      final hasPermission = await _recorder.hasPermission();

      if (!hasPermission) {
        if (!mounted) return;
        setState(() {
          status = 'مجوز میکروفون داده نشده است';
        });
        return;
      }

      final audioStream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 16000,
          numChannels: 1,
        ),
      );

      if (!mounted) {
        await _recorder.stop();
        return;
      }

      setState(() {
        isRunning = true;
        status = 'میکروفون فعال است؛ دریافت صدا...';
        _receivedBytes = 0;
      });

      _audioSubscription = audioStream.listen(
        (data) {
          _receivedBytes += data.length;

          const fiveSecondsBytes = 16000 * 2 * 5;

          if (_receivedBytes >= fiveSecondsBytes) {
            _receivedBytes -= fiveSecondsBytes;

            if (mounted && isRunning) {
              setState(() {
                status = '۵ ثانیه صدا دریافت شد؛ شنود ادامه دارد';
              });
            }
          }
        },
        onError: (error) {
          if (mounted) {
            setState(() {
              isRunning = false;
              status = 'خطا در دریافت صدای میکروفون';
            });
          }
        },
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          isRunning = false;
          status = 'شروع میکروفون ناموفق بود';
        });
      }
    }
  }

  Future<void> stopListening() async {
    await _audioSubscription?.cancel();
    _audioSubscription = null;

    try {
      await _recorder.stop();
    } catch (_) {}

    if (mounted) {
      setState(() {
        isRunning = false;
        status = isPlaying ? 'در حال پخش صدا' : 'برنامه متوقف است';
        _receivedBytes = 0;
      });
    }
  }

  Future<void> playRandomSound() async {
    if (isPlaying) return;

    // وضعیت شنود را پیش از شروع پخش ذخیره می‌کنیم.
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

      // رویداد پایان پخش را قبل از شروع پخش ثبت می‌کنیم.
      _completeSubscription = _player.onPlayerComplete.listen((_) async {
        await _finishPlayback('پخش صدا تمام شد');
      });

      await _player.play(AssetSource('sounds/$number.wav'));
    } catch (e) {
      await _finishPlayback('پخش صدا ناموفق بود');
    }
  }

  Future<void> stopPlayback() async {
    if (!isPlaying) return;

    // وضعیت قبلی شنود حفظ می‌شود و پس از قطع صدا بررسی خواهد شد.
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

    // فقط اگر پیش از پخش شنود فعال بوده، آن را خودکار شروع می‌کنیم.
    if (shouldResume && mounted) {
      await startListening();
    }
  }

  Future<void> exitApp() async {
    if (isRunning) {
      await stopListening();
    }

    if (isPlaying) {
      await _player.stop();
      await _completeSubscription?.cancel();
      _completeSubscription = null;
    }

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
      await SystemNavigator.pop();
    }
  }

  @override
  void dispose() {
    _audioSubscription?.cancel();
    _completeSubscription?.cancel();
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
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                isPlaying
                    ? Icons.volume_up
                    : isRunning
                        ? Icons.hearing
                        : Icons.hearing_disabled,
                size: 80,
                color: isPlaying
                    ? Colors.orange
                    : isRunning
                        ? Colors.green
                        : Colors.grey,
              ),
              const SizedBox(height: 24),
              Text(
                status,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 18),
              ),
              const SizedBox(height: 32),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: isRunning || isPlaying ? null : startListening,
                  child: const Text('شروع شنود'),
                ),
              ),
              const SizedBox(height: 12),

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
              const SizedBox(height: 12),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: isPlaying ? null : playRandomSound,
                  child: const Text('پخش تصادفی'),
                ),
              ),
              const SizedBox(height: 12),

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
              const SizedBox(height: 12),

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
