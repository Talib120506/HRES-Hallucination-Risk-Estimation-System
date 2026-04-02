import { useState } from 'react'
import VerdictCard from './VerdictCard'

function ResultsPanel({ results }) {
  const [expandedSection, setExpandedSection] = useState(null)

  if (!results) {
    return null
  }

  const toggleSection = (section) => {
    setExpandedSection(expandedSection === section ? null : section)
  }

  // Calculate combined verdict
  const getCombinedVerdict = () => {
    const { whitebox, blackbox, whitebox_error, blackbox_error } = results

    if (whitebox_error && blackbox_error) {
      return { label: 'ERROR', message: 'Both pipelines failed' }
    }

    let whiteboxLabel = null
    let blackboxLabel = null

    // Get whitebox label (SVM or XGBoost)
    if (whitebox?.SVM) {
      whiteboxLabel = whitebox.SVM.label
    } else if (whitebox?.XGBoost) {
      whiteboxLabel = whitebox.XGBoost.label
    }

    // Get blackbox label
    if (blackbox) {
      blackboxLabel = blackbox.verdict
    }

    // Agreement logic
    if (whiteboxLabel && blackboxLabel) {
      const whiteboxHallucinated = whiteboxLabel === 'HALLUCINATED'
      const blackboxHallucinated = blackboxLabel === 'HALLUCINATION'

      if (whiteboxHallucinated === blackboxHallucinated) {
        return {
          label: whiteboxHallucinated ? 'HALLUCINATION' : 'CORRECT',
          message: 'Both pipelines agree',
          confidence: 'HIGH',
        }
      } else {
        return {
          label: 'UNCERTAIN',
          message: 'Pipelines disagree - requires manual review',
          confidence: 'LOW',
        }
      }
    } else if (whiteboxLabel || blackboxLabel) {
      return {
        label: 'UNCERTAIN',
        message: 'Only one pipeline succeeded',
        confidence: 'MEDIUM',
      }
    }

    return { label: 'UNKNOWN', message: 'No results available' }
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

      {/* Combined Verdict */}
      <div className="transform transition-all">
        <button
          onClick={() => toggleSection('combined')}
          className="w-full flex items-center justify-between p-5 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border border-purple-100 dark:border-purple-800/30 rounded-xl hover:shadow-md transition-all duration-300 group"
        >
          <span className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
            <span className="text-2xl group-hover:scale-110 transition-transform">🎯</span> 
            Combined Verdict
          </span>
          <span className="text-gray-600 dark:text-gray-400 bg-white/50 dark:bg-black/20 w-8 h-8 rounded-full flex items-center justify-center transition-transform duration-300">
            {expandedSection === 'combined' ? (
               <svg className="w-5 h-5 transform rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
            ) : (
               <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
            )}
          </span>
        </button>
        {expandedSection === 'combined' && (
          <div className="mt-4 animate-[fadeInUp_0.3s_ease-out_forwards]">
            <VerdictCard
              type="Combined Analysis"
              label={combinedVerdict.label}
              details={{ message: combinedVerdict.message }}
            />
            <div className="mt-4 p-5 bg-gray-50/50 dark:bg-[#0f1016]/50 border border-gray-100 dark:border-gray-800 rounded-xl backdrop-blur-sm">
              <p className="text-sm text-gray-700 dark:text-gray-300 flex items-center gap-2">
                <span className="font-semibold px-2 py-1 bg-white dark:bg-gray-800 rounded-md shadow-sm">Confidence:</span>
                <span className={combinedVerdict.confidence === 'HIGH' ? 'text-green-600 dark:text-green-400 font-bold' : combinedVerdict.confidence === 'LOW' ? 'text-red-500 font-bold' : 'text-yellow-600 font-bold'}>{combinedVerdict.confidence || 'N/A'}</span>
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-3 leading-relaxed border-t border-gray-200 dark:border-gray-700 pt-3">{combinedVerdict.message}</p>
            </div>
          </div>
        )}
      </div>

      {/* Whitebox Details */}
      <div className="transform transition-all">
        <button
          onClick={() => toggleSection('whitebox')}
          className="w-full flex items-center justify-between p-5 bg-purple-50/50 dark:bg-purple-900/10 border border-purple-100/50 dark:border-purple-800/20 rounded-xl hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-all duration-300 group"
        >
          <span className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
            <span className="text-2xl group-hover:scale-110 transition-transform">🔬</span> 
            Whitebox Details (HRES)
          </span>
          <span className="text-gray-600 dark:text-gray-400 bg-white/50 dark:bg-black/20 w-8 h-8 rounded-full flex items-center justify-center transition-transform duration-300">
             {expandedSection === 'whitebox' ? (
                 <svg className="w-5 h-5 transform rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
              ) : (
                 <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
              )}
          </span>
        </button>
        {expandedSection === 'whitebox' && (
          <div className="mt-4 space-y-4 animate-[fadeInUp_0.3s_ease-out_forwards]">
            {results.whitebox_error ? (
              <div className="bg-red-50/80 dark:bg-red-900/20 border border-red-300 dark:border-red-800/50 rounded-xl p-4 text-red-700 dark:text-red-400">
                ❌ {results.whitebox_error}
              </div>
            ) : results.whitebox ? (
              <>
                {results.whitebox.SVM && (
                  <VerdictCard
                    type="SVM Classifier"
                    label={results.whitebox.SVM.label}
                    confidence={results.whitebox.SVM.confidence}
                    details={{
                      prob_correct: results.whitebox.SVM.prob_correct,
                      prob_hallucinated: results.whitebox.SVM.prob_hallucinated,
                    }}
                  />
                )}
                {results.whitebox.XGBoost && (
                  <VerdictCard
                    type="XGBoost Classifier"
                    label={results.whitebox.XGBoost.label}
                    confidence={results.whitebox.XGBoost.confidence}
                    details={{
                      prob_correct: results.whitebox.XGBoost.prob_correct,
                      prob_hallucinated: results.whitebox.XGBoost.prob_hallucinated,
                    }}
                  />
                )}
              </>
            ) : (
              <div className="text-gray-500 dark:text-gray-400 italic p-4 text-center bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-dashed border-gray-200 dark:border-gray-700">No whitebox results</div>
            )}
          </div>
        )}
      </div>

      {/* Blackbox Details */}
      <div className="transform transition-all">
        <button
          onClick={() => toggleSection('blackbox')}
          className="w-full flex items-center justify-between p-5 bg-blue-50/50 dark:bg-blue-900/10 border border-blue-100/50 dark:border-blue-800/20 rounded-xl hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all duration-300 group"
        >
          <span className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
            <span className="text-2xl group-hover:scale-110 transition-transform">🔍</span> 
            Blackbox Details (NLI)
          </span>
          <span className="text-gray-600 dark:text-gray-400 bg-white/50 dark:bg-black/20 w-8 h-8 rounded-full flex items-center justify-center transition-transform duration-300">
             {expandedSection === 'blackbox' ? (
                 <svg className="w-5 h-5 transform rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
              ) : (
                 <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
              )}
          </span>
        </button>
        {expandedSection === 'blackbox' && (
          <div className="mt-4 animate-[fadeInUp_0.3s_ease-out_forwards]">
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
        )}
      </div>
    </div>
  )
}

export default ResultsPanel
