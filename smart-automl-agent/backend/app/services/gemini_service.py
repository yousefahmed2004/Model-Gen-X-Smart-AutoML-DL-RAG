from __future__ import annotations
import logging
import re
import httpx

logger = logging.getLogger(__name__)

GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_KEY   = "AQ.Ab8RN6JzTn4ocH61JqVB3wRncHRPK-pGa7pM2RyCwh4JIQVqpg"   # ← ضع مفتاحك هنا

SYSTEM_PROMPT = """You are **Genix 🤖**, the AI training assistant for the **Model Gen X** platform. Your role is to guide users through the complete model training setup — step by step, like a knowledgeable and friendly expert, not a robotic form.

---

## 🌍 LANGUAGE DETECTION — ABSOLUTE RULE
- Detect the user's language from their FIRST message and maintain it for the ENTIRE conversation.
- Arabic first message → respond 100% in Arabic only. Zero English sentences.
- English first message → respond 100% in English only. Zero Arabic sentences.
- Technical terms (MLP, ResNet, SVM, BiLSTM, CSV, epoch, augmentation, etc.) are always acceptable in both languages.
- If the user switches language mid-conversation → switch immediately and permanently.
- NEVER mix languages in the same sentence or paragraph.

---

## 🎭 PERSONALITY & TONE
- Warm, encouraging, professional, and occasionally playful.
- Celebrate good choices: "Great choice! 🎉", "Perfect! ✨", "Excellent instinct! 💡"
- Gently correct mistakes: "Oops! 😄 That model belongs to ML — for Deep Learning we have better options."
- Keep responses concise and focused — one step at a time. Never ask two things at once.
- Use emojis naturally and sparingly — warmth, not noise.

---

## 🧠 PLATFORM MODEL CATALOG

### Machine Learning (ML) — Traditional Models:
| Model             | Best For                                      |
|-------------------|-----------------------------------------------|
| Auto (Best)       | Tries all ML models, picks the top performer 🏆 |
| Random Forest     | Robust all-rounder, handles noise well 🌲       |
| Gradient Boosting | High accuracy (similar to XGBoost) 💥           |
| SVM               | Small/medium datasets, excellent accuracy 🎯    |
| KNN               | Simple, intuitive, good baseline 👥             |
| Decision Tree     | Interpretable, easy to explain 🌳               |
| Extra Trees       | Faster variant of Random Forest ⚡              |
| AdaBoost          | Boosts weak learners iteratively 🔗             |
| Logistic Regression | Fast, classification ONLY 🚀                  |

### Deep Learning (DL) — PyTorch Models:
| Model        | Best For                                   |
|--------------|--------------------------------------------|
| MLP          | Tabular/structured data (CSV, Excel) 📊     |
| ResNet-18    | Image classification, general-purpose 🖼️   |
| ResNet-50    | Image classification, higher accuracy 🖼️   |
| MobileNetV2  | Image classification, lightweight & fast 📱 |
| BiLSTM       | Text, sequences, time-series 📝             |

### ⚠️ CRITICAL MODEL RULES:
- ML models → ONLY available under Machine Learning.
- DL models → ONLY available under Deep Learning.
- If user picks ML model in DL mode or vice versa → correct gently and offer valid alternatives.
- Logistic Regression + regression task → warn and suggest alternative (Gradient Boosting, Random Forest).
- SVM for DL → "Oops! SVM is ML-only. For DL: MLP (tabular 📊), ResNet (images 🖼️), BiLSTM (text 📝)."

---

## 🗺️ CONVERSATION FLOW — ONE QUESTION PER MESSAGE, NEVER SKIP STEPS

### STEP 1 — Training Type
Ask: "Would you like to train a **Machine Learning** model or a **Deep Learning** model? 🤔"

### STEP 2 — Task Type
Ask: "What kind of task is this? **Classification** (predicting a category/label) or **Regression** (predicting a numeric value)? 🎯"

### STEP 3 — Target Column
Show the user their dataset columns and suggest the best one.
- ✅ PREFER: target, label, category, class, type, status, result, outcome, price, salary, score, sales, grade, diagnosis, churn, fraud, survived
- ❌ AVOID: id, _id, date, time, created_at, updated_at, index, uuid, timestamp, row_number
Say: "Here are your dataset columns: [list]. I'd suggest **[COLUMN_NAME]** as the target — it looks like the best fit! 🎯 Does that work?"
- For BiLSTM → ask: "Which column contains the text data? 📝"
- For ResNet/MobileNetV2 → ask: "Which column contains the image file paths? 🖼️"
- NEVER use "text", "image", or any placeholder as target_column — always the ACTUAL column name.

### STEP 4 — Model Selection
Show models available ONLY for the chosen training_type.
Ask: "Which model would you like to use? Or should I **pick the best one** for you? 🧠"
- User says "you choose" / "auto" / "اختار انت" / "الأفضل" → Auto (Best) for ML; infer best DL model from context.

### STEP 5 — Pre-Processing
Ask: "Now let's configure **data pre-processing**. Should I handle it **automatically** with smart defaults, or would you like to configure it **manually** step by step? ⚙️"

#### If AUTO → apply these defaults silently, then summarize in one line:
- Missing values: median (numeric), mode (categorical)
- Encoding: One-Hot for ML, Label Encoding for DL
- Scaling: StandardScaler for SVM/LR/MLP; none for tree models
- Feature selection: disabled
- Train/test split: 80/20

#### If MANUAL → ask each sub-question separately, in order:

**5a. Missing Values**
"How should I handle **missing values**?
  1️⃣ Fill with mean/median (numeric) and mode (categorical) — recommended
  2️⃣ Drop rows with missing values
  3️⃣ Fill with a custom value
  4️⃣ Leave as-is (not recommended)"

**5b. Categorical Encoding**
"How should I encode **categorical (text) features**?
  1️⃣ One-Hot Encoding — best for ML algorithms
  2️⃣ Label Encoding — useful for tree models and DL
  3️⃣ Auto-detect and apply the best method"

**5c. Feature Scaling**
"Should I **scale** the numeric features?
  1️⃣ StandardScaler (mean=0, std=1) — best for SVM, Logistic Regression, MLP
  2️⃣ MinMaxScaler (range 0–1) — good for neural networks
  3️⃣ No scaling — fine for tree-based models
  4️⃣ Auto-select based on model"

**5d. Feature Selection** *(ask only if user seems engaged, skip if impatient)*
"Would you like to **remove irrelevant features** to improve performance?
  1️⃣ Yes — auto-select top features
  2️⃣ No — use all columns"

**5e. Train/Test Split**
"What **train/test split** ratio would you like?
  1️⃣ 80/20 — recommended ✅
  2️⃣ 70/30
  3️⃣ 90/10
  4️⃣ Custom ratio"
  ⚠️ If test split > 50% → warn: "A test split above 50% leaves very little training data — are you sure? 🤔"

### STEP 6 — Deep Learning Configuration (DL only)

**6a. Epochs**
"How many **training epochs**?
  1️⃣ 10 — quick test
  2️⃣ 50 — recommended ✅
  3️⃣ 100 — thorough training
  4️⃣ Custom"

**6b. Batch Size**
"What **batch size**?
  1️⃣ 16 — smaller, more updates per epoch
  2️⃣ 32 — recommended ✅
  3️⃣ 64 — faster, needs more memory
  4️⃣ Custom"

**6c. Optimizer**
"Which **optimizer**?
  1️⃣ Adam — fast and reliable, recommended ⚡
  2️⃣ SGD — classic, often better final accuracy
  3️⃣ RMSprop — great for RNNs and BiLSTM
  4️⃣ Auto-select"

**6d. Data Augmentation** *(ONLY for ResNet / MobileNetV2)*
"Would you like **data augmentation** on your images?
  1️⃣ Standard — flip, rotate, zoom, brightness adjustments
  2️⃣ Heavy — more aggressive transformations
  3️⃣ No augmentation"

### STEP 7 — Confirmation & Launch
Present a clean summary:

━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Training Configuration Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 Training Type:     [ML / Deep Learning]
🎯 Task:              [Classification / Regression]
📌 Target Column:     [column_name]
🤖 Model:             [model_name]
⚙️  Pre-processing:    [Auto / summary of manual choices]
📊 Train/Test Split:  [ratio]
[DL only below]
🔄 Epochs:            [n]
📦 Batch Size:        [n]
⚡ Optimizer:         [name]
🖼️  Augmentation:      [Yes/No — images only]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Then ask: "Everything look good? **Ready to start training?** 🚀"
Accept: "yes", "start", "go", "let's go", "ok", "تمام", "يلا", "ابدأ", "اه", "نعم" → OUTPUT JSON IMMEDIATELY.
If user wants to change something → loop back to that step, carry over all other confirmed answers.

---

## 📤 JSON OUTPUT — ONLY after all info collected AND user confirms

Replace EVERY value with what the user actually confirmed. Output this JSON block:
```json
{
  "action": "train",
  "training_type": "ml",
  "task_type": "classification",
  "target_column": "ACTUAL_COLUMN_NAME",
  "selected_model": "ACTUAL_MODEL_THE_USER_CHOSE",
  "preprocessing": {
    "missing_values": "median_mode",
    "encoding": "one_hot",
    "scaling": "standard",
    "feature_selection": false,
    "train_test_split": 0.2
  },
  "dl_config": {
    "epochs": 50,
    "batch_size": 32,
    "augmentation": false,
    "optimizer": "adam"
  }
}
```

### ⚠️ CRITICAL JSON RULES — VIOLATING THESE IS A SERIOUS ERROR:
- `selected_model` MUST be the EXACT model the user picked in STEP 4:
  - User said "KNN"            → `"selected_model": "KNN"` ✅  NOT "Auto (Best)" ❌
  - User said "Random Forest"  → `"selected_model": "Random Forest"` ✅
  - User said "SVM"            → `"selected_model": "SVM"` ✅
  - User said "اختار انت" / "auto" / "الأفضل" → `"selected_model": "Auto (Best)"` ✅
- `target_column` MUST be the exact column the user confirmed in STEP 3. NEVER a placeholder.
- `task_type` → "classification" or "regression" ONLY — never "auto".
- `missing_values` → "median_mode" | "drop" | "custom" | "none"
- `encoding` → "one_hot" | "label" | "auto"
- `scaling` → "standard" | "minmax" | "none" | "auto"
- `feature_selection` → true | false
- `train_test_split` → decimal 0.1–0.5 (e.g. 0.2 for 80/20, 0.3 for 70/30)
- Include `dl_config` ONLY for DL training_type. Omit entirely for ML.
- Include `augmentation` inside `dl_config` ONLY for ResNet / MobileNetV2.

---

## 🔁 EDGE CASES

- **Off-topic message during flow** → answer in 1–2 sentences max, then: "Let's get back to it — [repeat current step]."
- **User seems overwhelmed** → "No worries! Let's take it one step at a time 😊. [Re-ask current step simply]"
- **User gives multiple answers at once** → accept all valid ones, continue from first unanswered step.
- **User asks what a model does** → 2-sentence plain explanation, then continue.
- **User wants to change a confirmed answer** → "No problem! 🔄 Let's go back and update that." → loop to that step.
- **epochs = 0 or negative** → "That doesn't seem right 😄 — epochs must be a positive number!"
- **BiLSTM + no visible text column** → ask user to identify which column has text.
- **ResNet/MobileNetV2 + no image path column** → ask user to identify the image path column.

---

## 🧩 INTENT RECOGNITION — FUZZY MATCHING

| User says | Interpreted as |
|---|---|
| "machine", "ml", "ماشين", "مشين", "traditional", "تقليدي" | training_type = "ml" |
| "deep", "dl", "ديب", "neural", "شبكة عصبية" | training_type = "dl" |
| "classification", "تصنيف", "classify", "labels" | task_type = "classification" |
| "regression", "انحدار", "predict a value", "قيمة" | task_type = "regression" |
| "auto", "best", "you choose", "اختار انت", "الأفضل" | selected_model = "Auto (Best)" for ML |
| "random forest", "rf", "forest" | selected_model = "Random Forest" |
| "gradient boosting", "gb", "xgboost", "boost" | selected_model = "Gradient Boosting" |
| "svm", "support vector" | selected_model = "SVM" |
| "knn", "k-nearest", "neighbors" | selected_model = "KNN" |
| "decision tree", "dt", "tree" | selected_model = "Decision Tree" |
| "extra trees", "et" | selected_model = "Extra Trees" |
| "adaboost", "ada" | selected_model = "AdaBoost" |
| "logistic", "logistic regression", "lr" | selected_model = "Logistic Regression" |
| "mlp", "tabular", "dense", "fully connected" | selected_model = "MLP" |
| "resnet", "resnet18", "resnet50", "cnn" | selected_model = "ResNet" |
| "mobilenet", "mobilenetv2", "mobile" | selected_model = "MobileNetV2" |
| "bilstm", "lstm", "text model", "rnn", "sequence" | selected_model = "BiLSTM" |
| "yes", "ok", "go", "start", "يلا", "ابدأ", "اه", "تمام", "نعم" | Confirm / proceed |
| "no", "change", "go back", "لا", "غير", "ارجع" | Undo / revise last step |
| "adam" | optimizer = "adam" |
| "sgd" | optimizer = "sgd" |
| "rmsprop", "rms" | optimizer = "rmsprop" |
| "80/20", "0.2" | train_test_split = 0.2 |
| "70/30", "0.3" | train_test_split = 0.3 |
| "10 epochs", "50 epochs", "100 epochs" | epochs = 10 / 50 / 100 |
| "batch 16", "batch 32", "batch 64" | batch_size = 16 / 32 / 64 |

---

## 🔒 ABSOLUTE RULES — NEVER VIOLATE

1. Never output JSON until the user explicitly confirms ("yes", "start", "go", etc.)
2. Never use "text", "image", or any placeholder as target_column — always the actual column name.
3. Never ask two questions in the same message.
4. Never skip a step unless the user already provided that info unprompted.
5. Never mix languages in the same response.
6. Never assign an ML model to DL training or vice versa.
7. Never recommend Logistic Regression for regression tasks.
8. Always carry forward all confirmed answers — never re-ask what was already confirmed.
9. Never expose field names, internal state, or the JSON structure before final output.
10. Always be encouraging — there are no wrong choices, only better-suited models.
"""


class GeminiService:
    def __init__(self):
        self.enabled = True
        self.model   = GEMINI_MODEL
        logger.info("Gemini AI ready: %s", GEMINI_MODEL)

    def _is_arabic(self, text: str) -> bool:
        return bool(re.search(r'[\u0600-\u06FF]', text))

    def _normalize(self, text: str) -> str:
        return re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text.lower()).strip()

    def _pick_target_column(self, ctx: dict) -> str | None:
        if not ctx or 'columns' not in ctx:
            return None
        cols = [c['name'] for c in ctx.get('columns', []) if c.get('name')]
        preferred = [
            'target', 'label', 'category', 'class', 'status', 'outcome', 'result',
            'survived', 'churn', 'fraud', 'diagnosis', 'grade',
            'price', 'salary', 'score', 'sales', 'value'
        ]
        for kw in preferred:
            for col in cols:
                if kw in col.lower():
                    return col
        for col in cols:
            if not re.search(r'\b(id|_id|date|time|created|index|uuid|timestamp|row)\b', col.lower()):
                return col
        return cols[0] if cols else None

    def _last_assistant_message(self, history: list[dict]) -> str:
        for h in reversed(history):
            if h.get('role') in ('assistant', 'model'):
                return h.get('content', '') or ''
        return ''

    def _detect_preprocessing_choice(self, lower: str) -> str | None:
        """Detect auto vs manual preprocessing intent."""
        if re.search(r'\b(auto|automatic|automatically|smart|best|اوتو|تلقائي|اختار انت)\b', lower):
            return 'auto'
        if re.search(r'\b(manual|manually|step by step|يدوي|خطوة)\b', lower):
            return 'manual'
        return None

    def _detect_split(self, text: str) -> float | None:
        """Detect train/test split ratio from user message (raw text, not lowercased)."""
        m = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', text)
        if m:
            train, test = int(m.group(1)), int(m.group(2))
            total = train + test
            if total in (100, 10):
                return (test / total)
        lower = text.lower()
        if re.search(r'0\.2\b|eighty.?twenty|ثمانين.?عشرين', lower):
            return 0.2
        if re.search(r'0\.3\b|seventy.?thirty|سبعين.?ثلاثين', lower):
            return 0.3
        if re.search(r'0\.1\b|ninety.?ten|تسعين.?عشرة', lower):
            return 0.1
        if re.search(r'^[1١]$', text.strip()):
            return 0.2
        if re.search(r'^[2٢]$', text.strip()):
            return 0.3
        if re.search(r'^[3٣]$', text.strip()):
            return 0.1
        return None

    def _handle_guided_answer(self, history, user_message, dataset_context):
        """
        Fast-path handler for simple, predictable answers.
        Returns a response string if we can handle it locally,
        otherwise returns None to fall through to the LLM.
        """
        if not history:
            return None
        last_assistant = self._last_assistant_message(history)
        if not last_assistant:
            return None

        normalized = self._normalize(user_message)
        lower = normalized.lower()
        if not lower:
            return None

        arabic = self._is_arabic(user_message) or self._is_arabic(last_assistant)

        # --- Detect user intents ---
        training_type_ml = bool(re.search(
            r'\b(ماشين|مشين|ml|machine|traditional|تقليدي)\b', lower))
        training_type_dl = bool(re.search(
            r'\b(ديب|dl|deep|شبكة عصبية|neural)\b', lower))
        task_classification = bool(re.search(
            r'\b(تصنيف|classification|classify|كلاسيفيكيشن|labels?|category)\b', lower))
        task_regression = bool(re.search(
            r'\b(انحدار|regression|value|number|predict|نبؤ|تنبو|price|numeric)\b', lower))
        confirmation = bool(re.search(
            r'(يلا|ابدأ|اه|نعم|تمام|موافق|ماشي|أكيد|صح|اوكي|'
            r'yes|ok|go|start|sure|correct|looks.?good|perfect|agree|fine|yep|yup)', lower))
        negation = bool(re.search(
            r'\b(لا|no|nope|change|ارجع|غير|back|different)\b', lower))
        preprocessing_choice = self._detect_preprocessing_choice(lower)
        split = self._detect_split(user_message)

        def col_list(ctx, n=8):
            return ', '.join(c['name'] for c in ctx.get('columns', [])[:n]) if ctx else ''

        # --- STEP 1: Training type question was just asked ---
        asked_training_type = bool(re.search(
            r'machine learning.*deep learning|ml.*dl|ماشين.*ديب|ديب.*ماشين',
            last_assistant, re.IGNORECASE))

        if asked_training_type:
            if training_type_ml:
                return ('تمام! ماشين لارنينج 🤖 تصنيف ولا تنبؤ بقيمة؟'
                        if arabic else
                        'Great! Machine Learning it is 🤖 — Classification or Regression? 🎯')
            if training_type_dl:
                return ('ممتاز! ديب لارنينج 🧠 تصنيف ولا تنبؤ بقيمة؟'
                        if arabic else
                        'Excellent! Deep Learning it is 🧠 — Classification or Regression? 🎯')

        # --- STEP 2: Task type question was just asked ---
        asked_task_type = bool(re.search(
            r'classification.*regression|تصنيف.*انحدار|regression.*classification|انحدار.*تصنيف',
            last_assistant, re.IGNORECASE))

        if asked_task_type and dataset_context:
            target = self._pick_target_column(dataset_context) or 'the most suitable column'
            cols = col_list(dataset_context)
            if task_classification or task_regression:
                if arabic:
                    return (f'تمام! 🎯 الأعمدة عندك: {cols}.\n'
                            f'أقترح عمود **{target}** كـ target — هو الأنسب! موافق؟ 😊')
                return (f'Perfect! Your columns are: {cols}.\n'
                        f'I suggest **{target}** as the target column — looks like the best fit! 🎯 Sound good?')

        # --- STEP 3: Target column confirmation ---
        asked_target_col = bool(re.search(
            r'أقترح عمود|suggest.*column|target column|sound good|موافق\?|مناسب\?|الأنسب|best fit|هو الأنسب',
            last_assistant, re.IGNORECASE))

        if asked_target_col:
            if confirmation and not negation:
                user_msgs = ' '.join(h.get('content', '') for h in history if h.get('role') == 'user')
                is_dl = bool(re.search(r'\b(dl|deep|ديب|deep learning|شبكة عصبية|neural)\b', user_msgs, re.IGNORECASE))
                if is_dl:
                    if arabic:
                        return ('عظيم! 🎉 في الديب لارنينج عندنا:\n'
                                '• **MLP** — للبيانات الجدولية 📊\n'
                                '• **ResNet-18 / ResNet-50** — للصور 🖼️\n'
                                '• **MobileNetV2** — للصور (أسرع وأخف) 📱\n'
                                '• **BiLSTM** — للنصوص والتسلسلات 📝\n\n'
                                'إيه الموديل اللي يناسب بياناتك؟')
                    return ('Great! 🎉 For Deep Learning we have:\n'
                            '• **MLP** — tabular data 📊\n'
                            '• **ResNet-18 / ResNet-50** — images 🖼️\n'
                            '• **MobileNetV2** — images, lightweight 📱\n'
                            '• **BiLSTM** — text and sequences 📝\n\n'
                            'Which one fits your data?')
                else:
                    if arabic:
                        return ('عظيم! 🎉 في الماشين لارنينج عندنا:\n'
                                '• **Auto (Best)** — يجرب كل الموديلات ويختار الأفضل 🏆\n'
                                '• **Random Forest** — قوي ومتوازن 🌲\n'
                                '• **Gradient Boosting** — دقة عالية 💥\n'
                                '• **SVM** — ممتاز للداتا الصغيرة 🎯\n'
                                '• **KNN**, **Decision Tree**, **Extra Trees**, **AdaBoost**, **Logistic Regression**\n\n'
                                'تحب أي واحد؟ أو أقول لك الأفضل؟')
                    return ('Great! 🎉 For Machine Learning we have:\n'
                            '• **Auto (Best)** — tries all models, picks the winner 🏆\n'
                            '• **Random Forest** — robust all-rounder 🌲\n'
                            '• **Gradient Boosting** — high accuracy 💥\n'
                            '• **SVM** — great for smaller datasets 🎯\n'
                            '• **KNN**, **Decision Tree**, **Extra Trees**, **AdaBoost**, **Logistic Regression**\n\n'
                            'Which model would you like, or should I pick the best one?')

        # --- STEP 5: Pre-processing question was just asked ---
        asked_preprocessing = bool(re.search(
            r'pre-?processing|preprocessing|automatically|manually|تجهيز البيانات|بريبروسيسينج',
            last_assistant, re.IGNORECASE))

        if asked_preprocessing and preprocessing_choice:
            if preprocessing_choice == 'auto':
                if arabic:
                    return ('تمام! ✅ هطبق الإعدادات الذكية:\n'
                            '• القيم الناقصة: median للأرقام، mode للنصوص\n'
                            '• الترميز: One-Hot للـ ML، Label Encoding للـ DL\n'
                            '• التطبيع: StandardScaler\n'
                            '• التقسيم: 80% تدريب / 20% اختبار\n\n'
                            'المعالجة المسبقة جاهزة! 🎉 هل تريد **feature selection** لتحسين الأداء؟\n'
                            '1️⃣ اه — اختار أفضل الأعمدة تلقائياً\n'
                            '2️⃣ لا — استخدم كل الأعمدة')
                return ('Got it! ✅ Applying smart defaults:\n'
                        '• Missing values: median (numeric), mode (categorical)\n'
                        '• Encoding: One-Hot for ML, Label Encoding for DL\n'
                        '• Scaling: StandardScaler\n'
                        '• Split: 80% train / 20% test\n\n'
                        'Pre-processing is all set! Would you like **feature selection** to improve performance?\n'
                        '1️⃣ Yes — auto-select top features\n'
                        '2️⃣ No — use all columns')
            if preprocessing_choice == 'manual':
                if arabic:
                    return ('أوكي! خطوة بخطوة 🔧\n\n'
                            '**الخطوة الأولى — القيم الناقصة:**\n'
                            '1️⃣ ملء بالـ mean/median (أرقام) والـ mode (نصوص) — موصى به\n'
                            '2️⃣ حذف الصفوف الناقصة\n'
                            '3️⃣ ملء بقيمة مخصصة\n'
                            '4️⃣ تركها كما هي (مش مستحسن)\n\n'
                            'تختار إيه؟')
                return ('Sure! Let\'s go step by step 🔧\n\n'
                        '**Step 1 — Missing Values:**\n'
                        '1️⃣ Fill with mean/median (numeric) and mode (categorical) — recommended\n'
                        '2️⃣ Drop rows with missing values\n'
                        '3️⃣ Fill with a custom value\n'
                        '4️⃣ Leave as-is (not recommended)\n\n'
                        'Which option do you prefer?')

        # --- STEP 5a: Missing values answer ---
        asked_missing = bool(re.search(
            r'missing val|قيم.*نقص|نقص.*قيم|missing values|القيم الناقصة|values',
            last_assistant, re.IGNORECASE))

        if asked_missing:
            if re.search(r'^[1١]$', user_message.strip()):
                nxt = ('تمام! ✅ سنملأ القيم الناقصة بالـ median/mode. الآن، كيف نشفر الأعمدة النصية؟\n'
                       '1️⃣ One-Hot Encoding — موصى به للـ ML\n'
                       '2️⃣ Label Encoding — للنماذج الشجرية\n'
                       '3️⃣ تلقائي' if arabic else
                       'Great! ✅ We\'ll fill missing values with median/mode. Now, how should I encode categorical columns?\n'
                       '1️⃣ One-Hot Encoding — best for ML\n'
                       '2️⃣ Label Encoding — for tree models\n'
                       '3️⃣ Auto-detect')
                return nxt
            if re.search(r'^[2٢]$', user_message.strip()):
                return ('تمام! ✅ سنحذف الصفوف الناقصة. كيف نشفر الأعمدة النصية؟\n'
                        '1️⃣ One-Hot Encoding\n2️⃣ Label Encoding\n3️⃣ تلقائي' if arabic else
                        'Got it! ✅ We\'ll drop rows with missing values. How should I encode categorical columns?\n'
                        '1️⃣ One-Hot Encoding\n2️⃣ Label Encoding\n3️⃣ Auto-detect')
            if re.search(r'median|mean|متوسط|mean.?mode|ملء.*متوسط|موصى', lower):
                return ('تمام! ✅ سنملأ القيم الناقصة بالـ median/mode. كيف نشفر الأعمدة النصية؟\n'
                        '1️⃣ One-Hot Encoding — موصى به للـ ML\n'
                        '2️⃣ Label Encoding — للنماذج الشجرية\n'
                        '3️⃣ تلقائي' if arabic else
                        'Perfect! ✅ We\'ll fill with median/mode. How should I encode categorical columns?\n'
                        '1️⃣ One-Hot Encoding — best for ML\n'
                        '2️⃣ Label Encoding — for tree models\n'
                        '3️⃣ Auto-detect')

        # --- STEP 5b: Encoding answer ---
        asked_encoding = bool(re.search(
            r'encod|تشفير|categorical|نصية.*تشفير|one.?hot|label encod',
            last_assistant, re.IGNORECASE))

        if asked_encoding:
            if re.search(r'^[1١]$', user_message.strip()) or re.search(r'one.?hot|onehot|واحد.*حار', lower):
                return ('ممتاز! ✅ One-Hot Encoding. الآن، هل نحجم القيم الرقمية؟\n'
                        '1️⃣ StandardScaler — موصى به لـ KNN وSVM وMLP\n'
                        '2️⃣ MinMaxScaler — للشبكات العصبونية\n'
                        '3️⃣ بدون تحجيم — للنماذج الشجرية\n'
                        '4️⃣ تلقائي' if arabic else
                        'Perfect! ✅ One-Hot Encoding. Now, should I scale numeric features?\n'
                        '1️⃣ StandardScaler — best for KNN, SVM, MLP\n'
                        '2️⃣ MinMaxScaler — for neural networks\n'
                        '3️⃣ No scaling — fine for tree models\n'
                        '4️⃣ Auto-select')
            if re.search(r'^[2٢]$', user_message.strip()) or re.search(r'label|ordinal', lower):
                return ('تمام! ✅ Label Encoding. الآن، هل نحجم القيم الرقمية؟\n'
                        '1️⃣ StandardScaler\n2️⃣ MinMaxScaler\n3️⃣ بدون تحجيم\n4️⃣ تلقائي' if arabic else
                        'Got it! ✅ Label Encoding. Now, should I scale numeric features?\n'
                        '1️⃣ StandardScaler\n2️⃣ MinMaxScaler\n3️⃣ No scaling\n4️⃣ Auto-select')
            if re.search(r'^[3٣]$', user_message.strip()) or re.search(r'تلقائي|auto', lower):
                return ('تمام! ✅ تشفير تلقائي. الآن، هل نحجم القيم الرقمية؟\n'
                        '1️⃣ StandardScaler\n2️⃣ MinMaxScaler\n3️⃣ بدون تحجيم\n4️⃣ تلقائي' if arabic else
                        'Got it! ✅ Auto encoding. Now, should I scale numeric features?\n'
                        '1️⃣ StandardScaler\n2️⃣ MinMaxScaler\n3️⃣ No scaling\n4️⃣ Auto-select')

        # --- STEP 5c: Scaling answer ---
        asked_scaling = bool(re.search(
            r'scal|تحجيم|تطبيع|standard|minmax|normaliz',
            last_assistant, re.IGNORECASE))

        if asked_scaling:
            auto_intent = bool(re.search(
                r'اعمل انت|اختار انت|المناسب|افضل|auto|best|تلقائي|^[4٤]$',
                user_message.strip(), re.IGNORECASE))
            standard_intent = bool(re.search(r'standard|^[1١]$', user_message.strip(), re.IGNORECASE))
            minmax_intent   = bool(re.search(r'minmax|min.?max|^[2٢]$', user_message.strip(), re.IGNORECASE))
            none_intent     = bool(re.search(r'no scal|بدون|none|^[3٣]$', user_message.strip(), re.IGNORECASE))

            if standard_intent or minmax_intent or none_intent or auto_intent:
                scale_label = ('StandardScaler' if standard_intent else
                               'MinMaxScaler' if minmax_intent else
                               'بدون تحجيم' if none_intent else 'تلقائي')
                scale_label_en = ('StandardScaler' if standard_intent else
                                  'MinMaxScaler' if minmax_intent else
                                  'No scaling' if none_intent else 'Auto')
                return (f'تمام! ✅ {scale_label}. هل تريد **feature selection** لتحسين الأداء؟\n'
                        '1️⃣ اه — اختار أفضل الأعمدة تلقائياً\n'
                        '2️⃣ لا — استخدم كل الأعمدة' if arabic else
                        f'Perfect! ✅ {scale_label_en}. Would you like **feature selection** to improve performance?\n'
                        '1️⃣ Yes — auto-select top features\n'
                        '2️⃣ No — use all columns')

        # --- STEP 5d: Feature selection answer ---
        asked_feature_sel = bool(re.search(
            r'feature select|اختيار.*ميزات|ميزات|feature.*select',
            last_assistant, re.IGNORECASE))

        if asked_feature_sel:
            auto_intent = bool(re.search(
                r'اعمل انت|اختار انت|المناسب|تلقائي|auto|yes|اه|نعم|^[1١]$',
                user_message.strip(), re.IGNORECASE))
            no_intent = bool(re.search(
                r'^[2٢]$|no|لا|all|كل', user_message.strip(), re.IGNORECASE))

            if auto_intent or no_intent:
                return (f'تمام! ✅ الآن، ما نسبة تقسيم التدريب/الاختبار؟\n'
                        '1️⃣ 80/20 — موصى به ✅\n'
                        '2️⃣ 70/30\n'
                        '3️⃣ 90/10\n'
                        '4️⃣ نسبة مخصصة' if arabic else
                        'Got it! ✅ Now, what train/test split ratio?\n'
                        '1️⃣ 80/20 — recommended ✅\n'
                        '2️⃣ 70/30\n'
                        '3️⃣ 90/10\n'
                        '4️⃣ Custom ratio')

        # --- STEP 5e: Split ratio was provided ---
        # Guard: skip this block entirely if the user is confirming/launching training
        is_launch_confirmation = bool(re.search(
            r'(يلا|ابدأ|ابدء|اه|نعم|تمام|موافق|ماشي|'
            r'yes|ok|go|start|sure|correct|looks.?good|perfect|agree|fine|yep|yup)',
            lower))

        asked_split = bool(re.search(
            r'train.*test|split|تقسيم|نسبة.*تدريب|تدريب.*اختبار|مجموعات تدريب|split ratio',
            last_assistant, re.IGNORECASE))

        if asked_split and split is not None and not is_launch_confirmation:
            if split > 0.5:
                warn = ('⚠️ تقسيم الاختبار أكتر من 50% — ده هيخلي بيانات التدريب قليلة جداً! متأكد؟ 🤔'
                        if arabic else
                        '⚠️ A test split above 50% leaves very little training data — are you sure? 🤔')
                return warn
            train_pct = int((1 - split) * 100)
            test_pct = int(split * 100)
            if arabic:
                return (f'ممتاز! {train_pct}/{test_pct} ✅ — كل الإعدادات جاهزة!\n'
                        f'اكتب **"يلا ابدأ"** لو مبسوط بالإعدادات! 🚀')
            return (f'Perfect! {train_pct}/{test_pct} split ✅ — All settings are ready!\n'
                    f'Type **"Start training"** when you\'re ready to launch! 🚀')

        # --- STEP 7: Summary was shown → user confirms → fall through to LLM for JSON ---
        # Detect if the last assistant message was the training summary
        asked_ready = bool(re.search(
            r'(جاهز.*تدريب|ready.*train|start.*train|يلا.*ابدأ|ابدأ.*التدريب|'
            r'everything.*good|looks.*good|كل.*جيد|هل.*جاهز|'
            r'ملخص.*إعدادات|Training Configuration Summary|جاهز لبدء التدريب)',
            last_assistant, re.IGNORECASE))

        # If summary was shown and user confirms → let LLM handle JSON output
        # (LLM has full context and will produce the correct JSON)
        if asked_ready and is_launch_confirmation:
            return None  # Fall through to LLM which will output the JSON

        # Cannot handle locally → fall through to LLM
        return None

    def chat(self, history, user_message, dataset_context=None):
        # Try fast-path handler first
        guided_reply = self._handle_guided_answer(history, user_message, dataset_context)
        if guided_reply:
            return guided_reply

        # ─── بناء الـ contents بصيغة Gemini ───
        contents = []

        # سياق الـ dataset كأول رسالة
        if dataset_context:
            ctx_lines = (
                "[DATASET INFO — use this throughout the conversation]\n"
                f"- File: {dataset_context.get('filename', 'unknown')}\n"
                f"- Rows: {dataset_context.get('n_rows', 0):,}\n"
                f"- Columns ({dataset_context.get('n_cols', 0)} total): "
                f"{', '.join(c['name'] for c in dataset_context.get('columns', [])[:15])}\n"
                f"- Suggested target: {dataset_context.get('target_column', 'unknown')}\n"
                "- Column dtypes: " + ", ".join(
                    f"{c['name']}({c.get('dtype', '?')})"
                    for c in dataset_context.get('columns', [])[:12]
                ) + "\n"
            )
            contents.append({
                "role": "user",
                "parts": [{"text": ctx_lines}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I have the dataset info."}]
            })

        # تاريخ المحادثة — آخر 14 رسالة
        for h in history[-14:]:
            role = h.get("role", "user")
            # Gemini يستخدم "model" بدل "assistant"
            gemini_role = "model" if role in ("assistant", "model") else "user"
            content = h.get("content", "")
            if content:
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })

        # رسالة المستخدم الحالية
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 600,
            }
        }

        try:
            r = httpx.post(
                f"{GEMINI_URL}?key={GEMINI_KEY}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            logger.error("Gemini error %s: %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.error("Gemini failed: %s", e)

        return self._fallback(user_message, dataset_context)

    def _fallback(self, msg, ctx):
        """Last-resort fallback — only reached when Gemini API itself fails."""
        arabic = self._is_arabic(msg)
        if arabic:
            return "عندي مشكلة في الاتصال دلوقتي. ممكن تعيد الرسالة تاني؟ 🙏"
        return "I'm having a connection issue right now. Could you please resend your message? 🙏"

    def status(self):
        return {"enabled": self.enabled, "model": self.model, "provider": "Gemini"}


gemini = GeminiService()