import VerdictCard from './VerdictCard'

function ResultsPanel({ results }) {
  if (!results) {
    return null
  }

  // Calculate combined verdict (matching app.py logic)
  const getCombinedVerdict = () => {
    const { whitebox, blackbox, whitebox_error, blackbox_error } = results

    if (whitebox_error && blackbox_error) {
      return { label: 'ERROR', message: 'Both pipelines failed. Please check your inputs.' }
    }

    let whiteboxLabel = null
    let whiteboxConf = 0
    let blackboxVerdict = null
    let blackboxEnt = 0

    // Get whitebox label from single classifier (new format from app.py)
    if (whitebox?.label) {
      whiteboxLabel = whitebox.label
      whiteboxConf = whitebox.confidence || 0
    }

    // Get blackbox verdict
    if (blackbox?.verdict) {
      blackboxVerdict = blackbox.verdict
      blackboxEnt = blackbox.entailment || 0
    }

    // Agreement logic (matching app.py exactly)
    if (whiteboxLabel && blackboxVerdict) {
      const wbSaysHall = whiteboxLabel === 'HALLUCINATED'
      const bbSaysHall = blackboxVerdict === 'HALLUCINATION'
      const bbSaysOk = blackboxVerdict === 'GROUNDED'

      if (wbSaysHall && bbSaysHall) {
        return {
          label: 'HALLUCINATION',
          message: 'Both pipelines agree: this answer is likely hallucinated.',
          confidence: 'HIGH',
        }
      } else if (!wbSaysHall && bbSaysOk) {
        return {
          label: 'CORRECT',
          message: 'Both pipelines agree: this answer appears grounded in the source.',
          confidence: 'HIGH',
        }
      } else if (wbSaysHall && !bbSaysHall) {
        if (blackboxVerdict === 'UNCERTAIN') {
          return {
            label: 'HALLUCINATION',
            message: 'Document lacks information to confirm the answer (NLI Neutral), and Whitebox flagged it as a hallucination.',
            confidence: 'MEDIUM',
          }
        } else {
          return {
            label: 'UNCERTAIN',
            message: 'Whitebox flagged hallucination, but NLI says grounded. Manual review recommended.',
            confidence: 'LOW',
          }
        }
      } else if (!wbSaysHall && bbSaysHall) {
        return {
          label: 'UNCERTAIN',
          message: 'NLI flagged hallucination, but whitebox says correct. Manual review recommended.',
          confidence: 'LOW',
        }
      } else {
        return {
          label: 'UNCERTAIN',
          message: 'Mixed signals from both pipelines. Manual verification recommended.',
          confidence: 'LOW',
        }
      }
    } else if (whiteboxLabel) {
      return {
        label: whiteboxLabel,
        message: 'Result from whitebox pipeline only (NLI unavailable).',
        confidence: 'MEDIUM',
      }
    } else if (blackboxVerdict) {
      return {
        label: blackboxVerdict,
        message: 'Result from blackbox pipeline only (whitebox unavailable).',
        confidence: 'MEDIUM',
      }
    }

    return { label: 'UNKNOWN', message: 'No results available', confidence: 'N/A' }
  }

  const combinedVerdict = getCombinedVerdict()

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 mb-6 pb-4 border-b border-gray-100 dark:border-gray-800">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center shadow-lg shadow-green-500/30">
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Analysis Complete</h2>
      </div>

      {/* Generated Answer Section - shown when AI generated the answer */}
      {results.generated_answer && (
        <div className="bg-purple-50/50 dark:bg-purple-900/10 border border-purple-100 dark:border-purple-800/30 rounded-xl p-4 mb-6 animate-[fadeInUp_0.3s_ease-out_forwards]">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">🤖</span>
            <h3 className="text-sm font-bold text-purple-700 dark:text-purple-300 uppercase tracking-wider">Generated Answer</h3>
          </div>
          <p className="text-gray-800 dark:text-gray-200 leading-relaxed">{results.generated_answer}</p>
        </div>
      )}

      {/* Whitebox & Blackbox simultaneous view */}
      <div className="grid grid-cols-1 gap-6 pb-6 border-b border-gray-100 dark:border-gray-800">
        {/* Whitebox Details */}
        <div className="bg-purple-50/30 dark:bg-purple-900/5 border border-purple-100/50 dark:border-purple-800/20 rounded-xl overflow-hidden animate-[fadeInUp_0.3s_ease-out_forwards]">
          <div className="p-4 bg-purple-50/80 dark:bg-purple-900/10 border-b border-purple-100/50 dark:border-purple-800/20 flex items-center gap-2">
            <span className="text-xl">🔬</span> 
            <span className="text-base font-bold text-gray-800 dark:text-gray-100">Whitebox Details (HRES)</span>
          </div>
          <div className="p-4 space-y-4">
            {results.whitebox_error ? (
              <div className="bg-red-50/80 dark:bg-red-900/20 border border-red-300 dark:border-red-800/50 rounded-xl p-4 text-red-700 dark:text-red-400">
                ❌ {results.whitebox_error}
              </div>
            ) : results.whitebox ? (
              <VerdictCard
                type={`${results.whitebox.model || 'Classifier'}`}
                label={results.whitebox.label}
                confidence={results.whitebox.confidence}
                details={{
                  prob_correct: results.whitebox.prob_correct,
                  prob_hallucinated: results.whitebox.prob_hallucinated,
                }}
              />
            ) : (
              <div className="text-gray-500 dark:text-gray-400 italic p-4 text-center bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-dashed border-gray-200 dark:border-gray-700">No whitebox results</div>
            )}
          </div>
        </div>

        {/* Blackbox Details */}
        <div className="bg-blue-50/30 dark:bg-blue-900/5 border border-blue-100/50 dark:border-blue-800/20 rounded-xl overflow-hidden animate-[fadeInUp_0.4s_ease-out_forwards]">
          <div className="p-4 bg-blue-50/80 dark:bg-blue-900/10 border-b border-blue-100/50 dark:border-blue-800/20 flex items-center gap-2">
            <span className="text-xl">🔍</span> 
            <span className="text-base font-bold text-gray-800 dark:text-gray-100">Blackbox Details (NLI)</span>
          </div>
          <div className="p-4">
            {results.blackbox_error ? (
              <div className="bg-red-50/80 dark:bg-red-900/20 border border-red-300 dark:border-red-800/50 rounded-xl p-4 text-red-700 dark:text-red-400">
                ❌ {results.blackbox_error}
              </div>
            ) : results.blackbox ? (
              <VerdictCard
                type="NLI Verdict"
                label={results.blackbox.verdict}
                details={{
                  entailment: results.blackbox.entailment,
                  neutral: results.blackbox.neutral,
                  contradiction: results.blackbox.contradiction,
                  retrieved_context: results.blackbox.retrieved_context,
                }}
              />
            ) : (
              <div className="text-gray-500 dark:text-gray-400 italic p-4 text-center bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-dashed border-gray-200 dark:border-gray-700">No blackbox results</div>
            )}
          </div>
        </div>
      </div>

      {/* Combined Verdict */}
      <div className="bg-gradient-to-br from-purple-50/50 to-blue-50/50 dark:from-purple-900/10 dark:to-blue-900/10 border border-purple-100 dark:border-purple-800/30 rounded-xl overflow-hidden animate-[fadeInUp_0.5s_ease-out_forwards]">
        <div className="p-4 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border-b border-purple-100/80 dark:border-purple-800/40 flex items-center gap-2">
          <span className="text-xl">🎯</span> 
          <span className="text-lg font-bold text-gray-800 dark:text-gray-100">Combined Verdict</span>
        </div>
        <div className="p-4">
          <VerdictCard
            type="Combined Analysis"
            label={combinedVerdict.label}
            details={{ message: combinedVerdict.message }}
          />
          <div className="mt-4 p-5 bg-white/60 dark:bg-[#0f1016]/60 border border-gray-100 dark:border-gray-800 rounded-xl backdrop-blur-sm">
            <p className="text-sm text-gray-700 dark:text-gray-300 flex items-center gap-2">
              <span className="font-semibold px-2 py-1 bg-white dark:bg-gray-800 rounded-md shadow-sm">Confidence:</span>
              <span className={combinedVerdict.confidence === 'HIGH' ? 'text-green-600 dark:text-green-400 font-bold' : combinedVerdict.confidence === 'LOW' ? 'text-red-500 font-bold' : 'text-yellow-600 font-bold'}>{combinedVerdict.confidence || 'N/A'}</span>
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-3 leading-relaxed border-t border-gray-200 dark:border-gray-700 pt-3">{combinedVerdict.message}</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultsPanel
