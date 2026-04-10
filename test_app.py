"""
Quick test script to verify the application works correctly
Tests the question: "Which watchOS version is this user guide based on?"
Expected: GROUNDED with high entailment
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.detection import blackbox_predict, whitebox_predict
from app.services.answer_generator import generate_answer

# Test configuration
PDF_PATH = os.path.join(os.path.dirname(__file__), "resources", "pdfs", "apple_watch.pdf")
QUESTION = "Which watchOS version is this user guide based on?"
EXPECTED_ANSWER = "watchOS 8.6"

print("="*80)
print("HRES Application Test")
print("="*80)
print(f"\nPDF: {os.path.basename(PDF_PATH)}")
print(f"Question: {QUESTION}")
print(f"Expected Answer: {EXPECTED_ANSWER}")
print("\n" + "="*80)

# Test 1: Generate Answer
print("\n[TEST 1] AI Answer Generation")
print("-"*80)
try:
    generated_answer, error = generate_answer(PDF_PATH, QUESTION)
    if error:
        print(f"❌ ERROR: {error}")
    else:
        print(f"✓ Generated Answer: {generated_answer}")
        print(f"  Length: {len(generated_answer)} chars")
except Exception as e:
    print(f"❌ EXCEPTION: {str(e)}")
    generated_answer = None

# Test 2: Blackbox (NLI) Detection - with expected answer
print("\n[TEST 2] Blackbox NLI Detection (Expected Answer)")
print("-"*80)
try:
    bb_result, bb_error = blackbox_predict(PDF_PATH, QUESTION, EXPECTED_ANSWER)
    if bb_error:
        print(f"❌ ERROR: {bb_error}")
    else:
        print(f"✓ Verdict: {bb_result['verdict']}")
        print(f"  Entailment: {bb_result['entailment']:.2%}")
        print(f"  Neutral: {bb_result['neutral']:.2%}")
        print(f"  Contradiction: {bb_result['contradiction']:.2%}")
        print(f"  Retrieved Context (first 200 chars): {bb_result['retrieved_context'][:200]}...")
        
        # Validation
        if bb_result['verdict'] == 'GROUNDED' and bb_result['entailment'] > 0.5:
            print(f"\n✅ PASS: Answer is GROUNDED with high entailment ({bb_result['entailment']:.2%})")
        else:
            print(f"\n❌ FAIL: Expected GROUNDED with high entailment, got {bb_result['verdict']} with {bb_result['entailment']:.2%}")
except Exception as e:
    print(f"❌ EXCEPTION: {str(e)}")
    import traceback
    traceback.print_exc()

# Test 3: Blackbox with generated answer (if available)
if generated_answer:
    print("\n[TEST 3] Blackbox NLI Detection (Generated Answer)")
    print("-"*80)
    try:
        bb_gen_result, bb_gen_error = blackbox_predict(PDF_PATH, QUESTION, generated_answer)
        if bb_gen_error:
            print(f"❌ ERROR: {bb_gen_error}")
        else:
            print(f"✓ Verdict: {bb_gen_result['verdict']}")
            print(f"  Entailment: {bb_gen_result['entailment']:.2%}")
            print(f"  Neutral: {bb_gen_result['neutral']:.2%}")
            print(f"  Contradiction: {bb_gen_result['contradiction']:.2%}")
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")

# Test 4: Whitebox Detection
print("\n[TEST 4] Whitebox Detection (Expected Answer)")
print("-"*80)
try:
    wb_result, wb_error = whitebox_predict(PDF_PATH, QUESTION, EXPECTED_ANSWER)
    if wb_error:
        print(f"❌ ERROR: {wb_error}")
    else:
        print(f"✓ Label: {wb_result['label']}")
        print(f"  Model: {wb_result['model']}")
        print(f"  Confidence: {wb_result['confidence']:.2%}")
        print(f"  P(Correct): {wb_result['prob_correct']:.2%}")
        print(f"  P(Hallucinated): {wb_result['prob_hallucinated']:.2%}")
except Exception as e:
    print(f"❌ EXCEPTION: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Test Complete")
print("="*80)
