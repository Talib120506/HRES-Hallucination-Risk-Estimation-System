# HRES Application - Ready to Run

## ✅ Status: READY FOR TESTING

### Test Configuration
- **Question**: "Which watchOS version is this user guide based on?"
- **Document**: apple_watch.pdf
- **Expected Answer**: watchOS 8.6
- **Expected Result**: GROUNDED with high entailment (>85%)

---

## 🚀 Quick Start

### Option 1: Automated Start & Test
```cmd
run_and_test.bat
```
This will:
1. Run validation tests
2. Start backend on port 8000
3. Start frontend on port 5173
4. Open http://localhost:5173

### Option 2: Manual Start
```cmd
# Terminal 1 - Backend
cd backend
python run.py

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

### Option 3: Test Only (No UI)
```cmd
env\Scripts\activate
python test_app.py
```

---

## 🧪 Testing the Application

### Test Case 1: AI Generate Answer Mode
1. Open http://localhost:5173
2. Click "Document Q&A Analyzer"
3. Select **"AI Generate Answer"** tab
4. Select document: **apple watch**
5. Enter question: **"Which watchOS version is this user guide based on?"**
6. Click **"Generate & Analyze Answer"**

**Expected Results:**
- ✅ Generated Answer: "watchOS 8.6" (or similar correct answer)
- ✅ Whitebox (HRES): CORRECT with high confidence
- ✅ Blackbox (NLI): GROUNDED with high entailment (>85%)
- ✅ Retrieved context contains watchOS version information
- ✅ Combined Verdict: CORRECT with HIGH confidence

### Test Case 2: Manual Answer Input Mode
1. Select **"Manual Answer Input"** tab
2. Select document: **apple watch**
3. Enter question: **"Which watchOS version is this user guide based on?"**
4. Enter answer: **"watchOS 8.6"**
5. Click **"Analyze Answer"**

**Expected Results:**
- ✅ Whitebox (HRES): CORRECT with high confidence
- ✅ Blackbox (NLI): GROUNDED with high entailment (>85%)
- ✅ Retrieved context: "Apple Watch User Guide Everything you need to know about Apple Watch watchOS 8.6"
- ✅ Combined Verdict: CORRECT with HIGH confidence

---

## 📝 Recent Changes

### Answer Generator Improvements
**File**: `backend/app/services/answer_generator.py`

1. **Increased Context Retrieval**
   - Changed from top 3 to top 5 chunks
   - Increased context window from 1600 to 2400 characters
   - Better coverage of relevant information

2. **Improved Answer Extraction**
   - Decodes both with and without special tokens
   - Extracts model response from `<start_of_turn>model` tags
   - Multiple fallback methods for robust extraction
   - Removes prompt contamination

3. **Debug Logging**
   - Prints retrieved context (question, chunks, character count)
   - Shows full generated text with tokens
   - Displays extracted answer for verification

### Blackbox NLI (Previously Fixed)
**Files**: `src/app.py`, `backend/app/services/detection.py`

1. **Sentence-Level NLI Checking**
   - Splits noisy chunks into sentences
   - Tests each sentence individually
   - Returns best entailment score

2. **Simplified Hypothesis Format**
   - Uses answer-only as hypothesis (not full Q&A format)
   - Prevents false positives from complex hypothesis structures

---

## 🔍 How It Works

### AI Generate Mode Flow
1. **Retrieve Context**: Gets top 5 chunks from PDF using question embedding
2. **Generate Answer**: Gemma-2-2b generates answer from retrieved context
3. **Extract Answer**: Parses Gemma's output to get clean answer
4. **Whitebox Pipeline**: Extracts Gemma hidden states → PCA → Logistic Regression
5. **Blackbox Pipeline**: Retrieves chunks → NLI entailment checking
6. **Combined Verdict**: Merges both pipeline results

### Manual Input Mode Flow
1. User provides question + answer
2. **Whitebox Pipeline**: (same as above)
3. **Blackbox Pipeline**: (same as above)
4. **Combined Verdict**: (same as above)

---

## 📊 Expected Performance

Based on previous testing (from data/results/nli_results.csv):
- **NLI Entailment**: 93.84%
- **Verdict**: GROUNDED
- **Retrieved Pages**: 1, 8, 7, 9, 16, 280
- **Retrieved Context**: "Apple Watch User Guide Everything you need to know about Apple Watch watchOS 8.6"

---

## 🐛 Troubleshooting

### Backend won't start
```cmd
cd backend
pip install -r requirements.txt
python run.py
```

### Frontend won't start
```cmd
cd frontend
npm install
npm run dev
```

### Models not loading
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check model paths in `backend/app/services/model_loader.py`
- Models should be in `models/` directory

### Generated answer shows full prompt
- Check backend terminal for debug output
- Should see "EXTRACTED ANSWER:" with clean text
- If not, the new extraction logic should fix this

---

## 📁 Key Files

- `backend/app/services/answer_generator.py` - AI answer generation
- `backend/app/services/detection.py` - Whitebox & Blackbox pipelines
- `backend/app/api/routes.py` - API endpoints
- `frontend/src/pages/AppPage.jsx` - Main UI
- `frontend/src/components/app/ResultsPanel.jsx` - Results display
- `src/app.py` - Original Gradio app (reference implementation)

---

## ✨ Ready to Test!

The application is configured and ready. The answer generator now:
- Retrieves more context (5 chunks instead of 3)
- Has better extraction logic to get clean answers
- Includes debug logging for troubleshooting

**Run `run_and_test.bat` to start testing!**
