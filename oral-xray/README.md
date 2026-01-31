# AI Features for Graduation Project

# Project Archeticture
```
my-ai-project/
├── data/                    # مكان تخزين البيانات (لا يُرفع على GitHub)
│   ├── raw/                 # البيانات الخام الأصلية (Immutable)
│   ├── processed/           # البيانات بعد التنظيف والتحضير
│   └── external/            # بيانات من مصادر خارجية
│
├── models/                  # الموديلات المدربة (Saved Models .pkl, .h5, .pt)
│   ├── feature1_model/      # تنظيم الموديلات حسب الفيتشر
│   └── feature2_model/
│
├── notebooks/               # للتجريب والاستكشاف (Jupyter Notebooks)
│   ├── 01_eda.ipynb         # تحليل البيانات الاستكشافي
│   ├── 02_feature1_exp.ipynb
│   └── ...
│
├── src/                     # الكود الأساسي للمشروع (Source Code)
│   ├── __init__.py
│   ├── config.py            # إعدادات المشروع (Paths, Hyperparameters)
│   ├── data_loader.py       # سكريبتات تحميل البيانات
│   ├── preprocessing.py     # دوال التنظيف والمعالجة المشتركة
│   │
│   ├── features/            # *** أهم جزء للـ 6 فيتشرز ***
│   │   ├── __init__.py
│   │   ├── feature_1.py     # لوجيك الفيتشر الأولى
│   │   ├── feature_2.py     # لوجيك الفيتشر الثانية
│   │   └── ...
│   │
│   └── utils.py             # دوال مساعدة عامة (Helper functions)
│
├── api/                     # (اختياري) لو هتعمل API بـ Flask/FastAPI
│   ├── main.py
│   └── schemas.py
│
├── tests/                   # اختبارات الكود (Unit Tests)
│
├── requirements.txt         # المكتبات المطلوبة (pip freeze > requirements.txt)
├── README.md                # شرح المشروع وكيفية تشغيله
└── .gitignore               # ملفات لا يجب رفعها (مثل data, models, .env)
```