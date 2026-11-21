# Developer Salary Predictor

## 1. Mission and problem

**Mission:** Bridging the gap between latent potential and realized ability by equipping young graduates with in-demand tech skills, preparing them for employment and economic sustainability.
**Problem:** Young graduates often do not know what their skills are worth. This project predicts a developer's yearly salary (USD) from experience, education, role, country, work arrangement and company size, so graduates can set realistic goals and negotiate fairly.

## 2. Dataset

The [Stack Overflow Annual Developer Survey 2024](https://github.com/StackExchange/Survey/blob/main/packages/archive/2024/results.csv) (65,437 responses, 114 columns). The target is `ConvertedCompYearly` (yearly pay in USD). After cleaning (salary kept between $3,000 and $500,000, ages 18-34, up to 15 years of professional experience, rows with a missing target removed) 12,848 rows remain. The model uses 8 inputs: `YearsCodePro`, `YearsCode`, `NumLanguages`, `EdLevel`, `DevType`, `Country`, `RemoteWork`, `OrgSize`. Four models were compared (SGD, Linear Regression, Decision Tree, Random Forest) and the best one is served by the API.

## 3. Public API (Swagger UI)

https://linear-regression-model-7hpc.onrender.com/docs

Routes: `POST /predict`, `POST /retrain`, `GET /docs`. The service runs on Render's free plan, so the first request after a quiet period can take up to a minute while it wakes up.

## 4. Video demo
https://youtu.be/QAs4tvqK76g

## 5. Run the mobile app

You need the [Flutter SDK](https://docs.flutter.dev/get-started/install) (3.10 or newer) and either an Android emulator, an iOS simulator or a phone with USB debugging on. Check your setup with `flutter doctor`.

1. Clone the repository and open the app folder:
   ```
   git clone https://github.com/ToluwaniOladeji/linear_regression_model.git
   cd linear_regression_model/summative/FlutterApp
   ```
2. Install the dependencies:
   ```
   flutter pub get
   ```
   If `http` is missing, run `flutter pub add http`.
3. Confirm the API address at the top of `lib/main.dart` is the public Render URL (no trailing slash):
   ```dart
   const String apiBaseUrl = 'https://linear-regression-model-7hpc.onrender.com';
   ```
4. Start an emulator (or plug in a phone) and check it is listed:
   ```
   flutter devices
   ```
5. Run the app:
   ```
   flutter run
   ```
6. Fill in the eight fields. The three number fields show their allowed range and warn you when you go over; the other five are searchable dropdowns. Tap **Predict**. The result, or any validation error from the API, appears in the box underneath.
7. If the first prediction is slow, wait up to a minute and try again: the free Render service is waking up.

## Repository layout

```
linear_regression_model/
├── README.md
└── summative/
    ├── pyproject.toml
    ├── uv.lock
    ├── linear_regression/multivariate.ipynb
    ├── API/        prediction.py, retrain.py, best_model.pkl, pyproject.toml, uv.lock
    └── FlutterApp/
```