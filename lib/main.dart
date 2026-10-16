
import 'package:flutter/material.dart';

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
  bool isRunning = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('برنامهٔ دفع کلاغ'),
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
                isRunning
                    ? Icons.hearing
                    : Icons.hearing_disabled,
                size: 80,
                color: isRunning
                    ? Colors.green
                    : Colors.grey,
              ),
              const SizedBox(height: 20),
              Text(
                isRunning
                    ? 'شنود فعال است'
                    : 'برنامه متوقف است',
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 30),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: isRunning
                      ? null
                      : () {
                          setState(() {
                            isRunning = true;
                          });
                        },
                  child: const Text('شروع شنود'),
                ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: isRunning
                      ? () {
                          setState(() {
                            isRunning = false;
                          });
                        }
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    foregroundColor: Colors.white,
                  ),
                  child: const Text('توقف شنود'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
