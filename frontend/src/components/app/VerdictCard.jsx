import { getVerdictIcon, getVerdictClasses, getBadgeClasses } from '../../utils/colors'

function VerdictCard({ type, label, confidence, details }) {
  return (
    <div className={`border-l-4 rounded-xl p-6 shadow-md backdrop-blur-sm ${getVerdictClasses(label)}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold">{type}</h3>
        <span className={`px-4 py-1.5 rounded-full text-sm font-semibold border ${getBadgeClasses(label)} shadow-sm`}>
          <span className="mr-1">{getVerdictIcon(label)}</span> {label}
        </span>
      </div>

      {confidence !== undefined && (
        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Confidence Score</span>
            <span className="text-lg font-bold">{(confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-800/80 rounded-full h-3 overflow-hidden border border-gray-300/50 dark:border-gray-700/50">
            <div
              className={`h-3 rounded-full transition-all duration-1000 ease-out ${
                label === 'CORRECT' || label === 'GROUNDED'
                  ? 'bg-gradient-to-r from-green-400 to-green-600'
                  : label === 'HALLUCINATED' || label === 'HALLUCINATION'
                  ? 'bg-gradient-to-r from-red-400 to-red-600'
                  : 'bg-gradient-to-r from-yellow-400 to-yellow-600'
              }`}
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
        </div>
      )}

      {details && (
        <div className="space-y-4 text-sm mt-6 border-t border-black/5 dark:border-white/5 pt-4">
          {details.prob_correct !== undefined && (
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="opacity-80">P(Correct)</span>
                <span className="font-semibold">{(details.prob_correct * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-800/80 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-green-500 h-2 rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${details.prob_correct * 100}%` }}
                />
              </div>
            </div>
          )}
          {details.prob_hallucinated !== undefined && (
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="opacity-80">P(Hallucinated)</span>
                <span className="font-semibold">{(details.prob_hallucinated * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-800/80 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-red-500 h-2 rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${details.prob_hallucinated * 100}%` }}
                />
              </div>
            </div>
          )}
          {details.entailment !== undefined && (
            <>
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="opacity-80">Entailment</span>
                  <span className="font-semibold">{(details.entailment * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-800/80 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all duration-1000 ease-out"
                    style={{ width: `${details.entailment * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="opacity-80">Neutral</span>
                  <span className="font-semibold">{(details.neutral * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-800/80 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-yellow-500 h-2 rounded-full transition-all duration-1000 ease-out"
                    style={{ width: `${details.neutral * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="opacity-80">Contradiction</span>
                  <span className="font-semibold">{(details.contradiction * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-800/80 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-red-500 h-2 rounded-full transition-all duration-1000 ease-out"
                    style={{ width: `${details.contradiction * 100}%` }}
                  />
                </div>
              </div>
            </>
          )}
          {details.retrieved_context && (
            <div className="mt-4 p-4 bg-white/50 dark:bg-black/20 rounded-lg border border-black/5 dark:border-white/5 shadow-inner">
              <p className="font-semibold mb-2 opacity-90">Retrieved Context:</p>
              <p className="opacity-80 text-xs italic leading-relaxed">{details.retrieved_context}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default VerdictCard
