
import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:record/record.dart';

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
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.green,
        ),
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

  StreamSubscription<Uint8List>? _audioSubscription;

  bool isRunning = false;
  String status = 'برنامه متوقف است';

  int _receivedBytes = 0;

  Future<void> startListening() async {
    if (isRunning) return;

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

          // هر ۵ ثانیه، دریافت صدا را گزارش می‌کنیم.
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
    } catch (_) {
      // آزاد کردن وضعیت برنامه حتی در صورت خطای توقف
    }

    if (mounted) {
      setState(() {
        isRunning = false;
        status = 'برنامه متوقف است';
        _receivedBytes = 0;
      });
    }
  }

  Future<void> exitApp() async {
    if (isRunning) {
      await stopListening();
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
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          'برنامهٔ دفع کلاغ',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
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
                isRunning ? Icons.hearing : Icons.hearing_disabled,
                size: 80,
                color: isRunning ? Colors.green : Colors.grey,
              ),
              const SizedBox(height: 24),
              Text(
                status,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 32),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: isRunning ? null : startListening,
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

