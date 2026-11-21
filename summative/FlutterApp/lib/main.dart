import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'options.dart'; // generated from api_schema.json by make_options.py

// Your public Render URL, with no trailing slash.
const String apiBaseUrl = 'https://linear-regression-model-7hpc.onrender.com';

void main() => runApp(const SalaryApp());

class SalaryApp extends StatelessWidget {
  const SalaryApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Developer Salary Predictor',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const PredictPage(),
    );
  }
}

enum FieldType { decimal, integer, text }

class FieldSpec {
  const FieldSpec(this.key, this.label, this.hint, this.type, {this.max, this.options});

  final String key; // must match the field name in the API's SalaryInput model
  final String label;
  final String hint;
  final FieldType type;
  final int? max; // number fields: the largest value the API accepts
  final List<String>? options; // fields with a fixed set of values: searchable dropdown

  // Number fields show their limit next to the name.
  String get displayLabel => max == null ? label : '$label (0 to $max)';
}

// One entry per model input (8 in total).
const List<FieldSpec> fieldSpecs = [
  FieldSpec('YearsCodePro', 'Years of professional coding', 'e.g. 3', FieldType.decimal,
      max: yearsCodeProMax),
  FieldSpec('YearsCode', 'Total years of coding', 'e.g. 5', FieldType.decimal, max: yearsCodeMax),
  FieldSpec('NumLanguages', 'Number of languages used', 'e.g. 4 (whole number)', FieldType.integer,
      max: numLanguagesMax),
  FieldSpec('EdLevel', 'Education level', 'Type to search or pick', FieldType.text,
      options: edLevelOptions),
  FieldSpec('DevType', 'Developer type', 'Type to search or pick', FieldType.text,
      options: devTypeOptions),
  FieldSpec('Country', 'Country', 'Type to search or pick', FieldType.text, options: countryOptions),
  FieldSpec('RemoteWork', 'Work arrangement', 'Type to search or pick', FieldType.text,
      options: remoteWorkOptions),
  FieldSpec('OrgSize', 'Company size', 'Type to search or pick', FieldType.text,
      options: orgSizeOptions),
];

class PredictPage extends StatefulWidget {
  const PredictPage({super.key});

  @override
  State<PredictPage> createState() => _PredictPageState();
}

class _PredictPageState extends State<PredictPage> {
  final Map<String, TextEditingController> _controllers = {
    for (final f in fieldSpecs) f.key: TextEditingController(),
  };

  bool _loading = false;
  String? _result;
  String? _error;

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  // Turns the text typed for a number field into a JSON number.
  // Text that is not a number is sent as text, so the API rejects it with a type error.
  Object _asNumber(String text, {required bool wholeOnly}) {
    final asInt = int.tryParse(text);
    if (asInt != null) return asInt;
    final asDouble = double.tryParse(text);
    if (asDouble != null && asDouble.isFinite) {
      if (wholeOnly && asDouble == asDouble.roundToDouble()) return asDouble.toInt(); // "4.0" -> 4
      return asDouble;
    }
    return text;
  }

  Map<String, dynamic> _buildBody() {
    final body = <String, dynamic>{};
    for (final f in fieldSpecs) {
      final text = _controllers[f.key]!.text.trim();
      if (text.isEmpty) continue; // left out, so the API answers "Field required"
      if (f.type == FieldType.text) {
        body[f.key] = text; // a dropdown choice, or free text the API will reject if unknown
      } else {
        body[f.key] = _asNumber(text, wholeOnly: f.type == FieldType.integer);
      }
    }
    return body;
  }

  // Live warning under a number field. It does not block the button: the API still has the final say.
  String? _limitWarning(FieldSpec f) {
    final max = f.max;
    if (max == null) return null;
    final text = _controllers[f.key]!.text.trim();
    if (text.isEmpty) return null;
    final n = num.tryParse(text);
    if (n == null || (n is double && !n.isFinite)) return 'Enter a number';
    if (f.type == FieldType.integer && n % 1 != 0) return 'Enter a whole number';
    if (n < 0) return 'Too low: the minimum is 0';
    if (n > max) return 'Too high: the maximum is $max';
    return null;
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    setState(() {
      _loading = true;
      _result = null;
      _error = null;
    });

    try {
      final response = await http
          .post(
            Uri.parse('$apiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(_buildBody()),
          )
          .timeout(const Duration(seconds: 70)); // a sleeping free Render service needs up to a minute

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        final value = (decoded['predicted_salary_usd'] as num).toDouble();
        _show(result: 'Predicted yearly salary\n${_money(value)}');
      } else if (response.statusCode == 422) {
        final decoded = jsonDecode(response.body);
        _show(error: _validationMessage(decoded['detail']));
      } else {
        _show(error: 'The server returned an error (${response.statusCode}). Please try again.');
      }
    } on TimeoutException {
      _show(error: 'The server took too long to respond. It may be waking up, so try again in a moment.');
    } on SocketException {
      _show(error: 'Could not reach the server. Check your internet connection.');
    } on http.ClientException {
      _show(error: 'Could not reach the server. Check your internet connection.');
    } on FormatException {
      _show(error: 'The server sent an unexpected reply. Please try again.');
    } catch (_) {
      _show(error: 'Something went wrong. Please try again.');
    }
  }

  void _show({String? result, String? error}) {
    if (!mounted) return;
    setState(() {
      _loading = false;
      _result = result;
      _error = error;
    });
  }

  String _money(double value) {
    final digits = value.abs().round().toString();
    final buffer = StringBuffer();
    for (var i = 0; i < digits.length; i++) {
      if (i > 0 && (digits.length - i) % 3 == 0) buffer.write(',');
      buffer.write(digits[i]);
    }
    final sign = value < 0 ? '-' : '';
    return '$sign\$$buffer';
  }

  // Turns FastAPI's validation errors into short, readable lines, one per problem field.
  String _validationMessage(dynamic detail) {
    if (detail is List) {
      final labels = {for (final f in fieldSpecs) f.key: f.label};
      final lines = <String>[];
      for (final item in detail) {
        final loc = ((item['loc'] as List?) ?? [])
            .where((p) => p != 'body')
            .map((p) => p.toString())
            .toList();
        final field = loc.isEmpty ? '' : (labels[loc.last] ?? loc.last);
        var msg = (item['msg'] ?? 'Invalid value').toString();
        if (msg.length > 120) msg = '${msg.substring(0, 117)}...';
        lines.add(field.isEmpty ? msg : '$field: $msg');
      }
      return lines.join('\n');
    }
    return detail.toString();
  }

  // Fixed-choice fields: a text box you can type in, with a menu that filters as you type.
  // Number fields: a plain text box with the limit in its label and a live warning.
  Widget _buildField(FieldSpec f) {
    final options = f.options;
    if (options != null) {
      return LayoutBuilder(
        builder: (context, constraints) => DropdownMenu<String>(
          controller: _controllers[f.key],
          width: constraints.maxWidth,
          label: Text(f.label),
          hintText: f.hint,
          enableFilter: true,
          requestFocusOnTap: true,
          menuHeight: 280,
          inputDecorationTheme: const InputDecorationTheme(border: OutlineInputBorder()),
          dropdownMenuEntries: [
            for (final option in options) DropdownMenuEntry<String>(value: option, label: option),
          ],
        ),
      );
    }
    return TextField(
      controller: _controllers[f.key],
      textInputAction: TextInputAction.next,
      onChanged: (_) => setState(() {}), // refresh the warning while typing
      decoration: InputDecoration(
        labelText: f.displayLabel,
        hintText: f.hint,
        errorText: _limitWarning(f),
        border: const OutlineInputBorder(),
      ),
    );
  }

  Widget _buildResultBox() {
    Color background = Colors.grey.shade200;
    Color border = Colors.grey.shade400;
    Widget content = const Text(
      'The predicted salary will appear here.',
      style: TextStyle(color: Colors.black54),
    );

    if (_loading) {
      content = const Row(
        children: [
          SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)),
          SizedBox(width: 12),
          Expanded(child: Text('Contacting the server. The first request can take up to a minute.')),
        ],
      );
    } else if (_result != null) {
      background = Colors.green.shade50;
      border = Colors.green.shade400;
      content = Text(
        _result!,
        style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
      );
    } else if (_error != null) {
      background = Colors.red.shade50;
      border = Colors.red.shade400;
      content = Text(_error!, style: TextStyle(color: Colors.red.shade900));
    }

    return Container(
      constraints: const BoxConstraints(minHeight: 96),
      padding: const EdgeInsets.all(16),
      alignment: Alignment.centerLeft,
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: border),
      ),
      child: content,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Developer Salary Predictor')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text('Enter a developer profile to estimate the yearly salary in USD.'),
              const SizedBox(height: 16),
              for (final f in fieldSpecs)
                Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: _buildField(f),
                ),
              const SizedBox(height: 4),
              SizedBox(
                height: 48,
                child: FilledButton(
                  onPressed: _loading ? null : _predict,
                  child: const Text('Predict'),
                ),
              ),
              const SizedBox(height: 16),
              _buildResultBox(),
            ],
          ),
        ),
      ),
    );
  }
}