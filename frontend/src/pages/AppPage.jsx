import { useState, useEffect } from 'react'
import Header from '../components/app/Header'
import TabContainer from '../components/app/TabContainer'
import PreloadedTab from '../components/app/PreloadedTab'
import UploadTab from '../components/app/UploadTab'
import ResultsPanel from '../components/app/ResultsPanel'
import { predictHallucination, generateAndVerify } from '../services/api'
import { useParallax } from '../hooks/useParallax'

function AppPage() {
  const bgOffset1 = useParallax(0.3);
  const bgOffset2 = useParallax(0.2);
  const bgOffset3 = useParallax(0.4);

  const [activeTab, setActiveTab] = useState('preloaded')
  const [mode, setMode] = useState('generate') // 'generate' or 'manual'
  const [formData, setFormData] = useState({
    file: null,
    preloaded_pdf: '',
    question: '',
    answer: '',
  })
  const [errors, setErrors] = useState({})
  const [loading, setLoading] = useState(false)
  const [loadingText, setLoadingText] = useState('Initializing analysis...')
  const [results, setResults] = useState(null)
  const [successMessage, setSuccessMessage] = useState('')

  useEffect(() => {
    if (loading) {
      const msgs = mode === 'generate' 
        ? [
            'Retrieving relevant context from document...',
            'Generating answer with Gemma...',
            'Running Whitebox pipeline...',
            'Validating through Blackbox NLI...',
            'Computing final hallucination verdict...'
          ]
        : [
            'Analyzing document context...',
            'Running Whitebox pipeline...',
            'Validating through Blackbox NLI...',
            'Cross-referencing features...',
            'Computing final hallucination verdict...'
          ]
      let i = 0
      setLoadingText(msgs[0])
      const interval = setInterval(() => {
        i = (i + 1) % msgs.length
        setLoadingText(msgs[i])
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [loading, mode])

  const validateForm = (isGenerateMode = false) => {
    const newErrors = {}

    // Validate PDF
    if (!formData.preloaded_pdf) {
      newErrors.pdf = 'Please select a document'
    }

    // Validate question
    if (!formData.question || !formData.question.trim()) {
      newErrors.question = 'Please enter a question'
    }

    // Validate answer only in manual mode
    if (!isGenerateMode && (!formData.answer || !formData.answer.trim())) {
      newErrors.answer = 'Please enter an answer to verify'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async () => {
    // Validate form (manual mode - needs answer)
    if (!validateForm(false)) {
      return
    }

    setResults(null)
    setSuccessMessage('')
    setLoading(true)

    try {
      const apiFormData = new FormData()
      apiFormData.append('question', formData.question)
      apiFormData.append('answer', formData.answer)
      apiFormData.append('preloaded_pdf', formData.preloaded_pdf)

      const response = await predictHallucination(apiFormData)
      setResults(response)

      setSuccessMessage('Analysis complete!')
      setTimeout(() => setSuccessMessage(''), 3000)

      setTimeout(() => {
        const resultsElement = document.getElementById('results-section')
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: 'smooth' })
        }
      }, 100)
    } catch (error) {
      setErrors({ submit: error.message })
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateSubmit = async () => {
    // Validate form (generate mode - no answer needed)
    if (!validateForm(true)) {
      return
    }

    setResults(null)
    setSuccessMessage('')
    setLoading(true)

    try {
      const apiFormData = new FormData()
      apiFormData.append('question', formData.question)
      apiFormData.append('preloaded_pdf', formData.preloaded_pdf)

      const response = await generateAndVerify(apiFormData)
      setResults(response)

      setSuccessMessage('Analysis complete!')
      setTimeout(() => setSuccessMessage(''), 3000)

      setTimeout(() => {
        const resultsElement = document.getElementById('results-section')
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: 'smooth' })
        }
      }, 100)
    } catch (error) {
      setErrors({ submit: error.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#0a0a0a] text-gray-900 dark:text-white transition-colors duration-300 relative overflow-hidden flex flex-col">
      {/* Background Decorative Blobs */}
      <div 
        className="fixed top-[-10%] -left-[10%] w-96 h-96 bg-purple-300/40 dark:bg-purple-900/30 rounded-full mix-blend-multiply filter blur-[100px] animate-blob z-0 pointer-events-none"
        style={{ transform: `translateY(${bgOffset1}px)` }}
      ></div>
      <div 
        className="fixed top-[20%] -right-[10%] w-96 h-96 bg-blue-300/40 dark:bg-blue-900/30 rounded-full mix-blend-multiply filter blur-[100px] animate-blob animation-delay-2000 z-0 pointer-events-none"
        style={{ transform: `translateY(${bgOffset2}px)` }}
      ></div>
      <div 
        className="fixed -bottom-[10%] left-[20%] w-96 h-96 bg-indigo-300/40 dark:bg-indigo-900/30 rounded-full mix-blend-multiply filter blur-[100px] animate-blob animation-delay-4000 z-0 pointer-events-none"
        style={{ transform: `translateY(${bgOffset3}px)` }}
      ></div>

      <Header />

      {/* Loading Overlay */}
      {loading && (
        <div className="fixed inset-0 bg-white/60 dark:bg-black/80 backdrop-blur-md z-50 flex items-center justify-center transition-all">
          <div className="bg-white/80 dark:bg-[#111111]/90 border border-gray-200/50 dark:border-gray-800/50 rounded-2xl p-10 shadow-2xl backdrop-blur-xl flex flex-col items-center max-w-sm w-[90%] animate-[fadeInUp_0.3s_ease-out_forwards]">
            {/* Custom Scanning Radar Animation */}
            <div className="relative w-24 h-24 mb-6">
              <div className="absolute inset-0 border-4 border-purple-200 dark:border-purple-900/30 rounded-full"></div>
              <div className="absolute inset-0 border-4 border-transparent border-t-purple-600 dark:border-t-purple-400 rounded-full animate-spin"></div>
              <div className="absolute inset-2 border-4 border-blue-200 dark:border-blue-900/30 rounded-full"></div>
              <div className="absolute inset-2 border-4 border-transparent border-b-blue-600 dark:border-b-blue-400 rounded-full animate-[spin_1.5s_linear_infinite_reverse]"></div>
              <div className="absolute inset-0 w-2 h-2 m-auto bg-purple-600 dark:bg-purple-400 rounded-full animate-ping"></div>
            </div>
            <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-blue-600 dark:from-purple-400 dark:to-blue-400 mb-2">
              {mode === 'generate' ? 'Generating & Analyzing' : 'Analyzing Data'}
            </h3>
            <p className="text-gray-600 dark:text-gray-300 font-medium text-center h-12 flex items-center justify-center animate-pulse">{loadingText}</p>
          </div>
        </div>
      )}

      {/* Success Toast */}
      {successMessage && (
        <div className="fixed top-24 right-4 bg-green-500/90 backdrop-blur-md text-white px-6 py-3 rounded-xl shadow-[0_4px_20px_rgba(34,197,94,0.4)] z-40 animate-[fadeInUp_0.3s_ease-out_forwards]">
          <span className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
            {successMessage}
          </span>
        </div>
      )}

      <main className="flex-grow container relative z-10 mx-auto max-w-7xl py-12 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel: Input Form */}
          <div className="group relative bg-white/70 dark:bg-[#111111]/70 backdrop-blur-xl border border-white/20 dark:border-gray-800/50 rounded-2xl shadow-xl overflow-hidden flex flex-col transform transition-all duration-500 hover:-translate-y-1 hover:shadow-[0_0_40px_rgba(124,58,237,0.3)] before:absolute before:inset-0 before:ring-2 before:ring-purple-500/0 hover:before:ring-purple-500/50 before:rounded-2xl before:transition-all before:duration-500">
            <div className="p-6 flex-grow relative z-10">
              <PreloadedTab
                formData={formData}
                setFormData={setFormData}
                onSubmit={handleSubmit}
                onGenerateSubmit={handleGenerateSubmit}
                errors={errors}
                mode={mode}
                setMode={setMode}
              />
              
              {errors.submit && (
                <div className="mt-6 bg-red-50/80 dark:bg-red-900/20 backdrop-blur-sm border border-red-300 dark:border-red-800/50 rounded-xl p-4 text-red-700 dark:text-red-400 flex items-start gap-3">
                  <svg className="w-5 h-5 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                  <span>{errors.submit}</span>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel: Results */}
          <div id="results-section" className="group relative bg-white/70 dark:bg-[#111111]/70 backdrop-blur-xl border border-white/20 dark:border-gray-800/50 rounded-2xl shadow-xl flex flex-col transform transition-all duration-500 hover:-translate-y-1 hover:shadow-[0_0_40px_rgba(59,130,246,0.3)] before:absolute before:inset-0 before:ring-2 before:ring-blue-500/0 hover:before:ring-blue-500/50 before:rounded-2xl before:transition-all before:duration-500 mt-0 pt-0">
            <div className="p-6 flex-grow relative z-10 w-full h-full flex flex-col">
            {results ? (
              <div className="animate-[fadeInUp_0.5s_ease-out_forwards]">
                <ResultsPanel results={results} />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center flex-grow min-h-[400px] text-gray-400 dark:text-gray-500">
                <div className="relative w-32 h-32 mb-6">
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-100 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/10 rounded-full animate-pulse"></div>
                  <svg
                    className="absolute inset-0 w-16 h-16 m-auto text-purple-300 dark:text-purple-700"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <p className="text-xl font-semibold text-gray-800 dark:text-gray-200">Awaiting Signal</p>
                <p className="text-sm mt-3 text-center max-w-xs leading-relaxed">
                  Configure parameters and run the analysis to view hallucination metrics here.
                </p>
              </div>
            )}
            </div>
          </div>
        </div>

        {/* Examples Section */}
        <div className="mt-12 bg-white/70 dark:bg-[#111111]/70 backdrop-blur-xl border border-gray-200/50 dark:border-gray-800/50 rounded-2xl shadow-lg p-6 hover:shadow-xl transition-all">
          <div className="flex items-center gap-2 mb-6">
            <span className="text-2xl">💡</span>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              Example Queries
            </h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div
              className="group cursor-pointer rounded-xl p-5 border border-gray-200 dark:border-gray-800 bg-gray-50/50 dark:bg-[#0a0a0a]/50 hover:bg-purple-50 dark:hover:bg-purple-900/10 hover:border-purple-300 dark:hover:border-purple-500/50 transition-all duration-300"
              onClick={() => {
                setMode('manual')
                setFormData({
                  ...formData,
                  preloaded_pdf: 'apple_watch.pdf',
                  question: 'Which watchOS version is this user guide based on?',
                  answer: 'watchOS 8.6',
                })
                setResults(null)
                window.scrollTo({ top: 0, behavior: 'smooth' })
              }}
            >
              <div className="flex items-start">
                <span className="text-2xl mr-3 group-hover:text-purple-500 transition-colors">✓</span>
                <div>
                  <p className="font-semibold text-gray-800 dark:text-gray-200 mb-2 group-hover:text-purple-600 dark:group-hover:text-purple-400">Correct Answer Example</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    A factual answer that is supported by the document
                  </p>
                </div>
              </div>
            </div>

            <div
              className="group cursor-pointer rounded-xl p-5 border border-gray-200 dark:border-gray-800 bg-gray-50/50 dark:bg-[#0a0a0a]/50 hover:bg-red-50 dark:hover:bg-red-900/10 hover:border-red-300 dark:hover:border-red-500/50 transition-all duration-300"
              onClick={() => {
                setMode('manual')
                setFormData({
                  ...formData,
                  preloaded_pdf: 'apple_watch.pdf',
                  question: 'What is the battery life of the Apple Watch?',
                  answer: 'The Apple Watch has a battery life of 72 hours with always-on display enabled.',
                })
                setResults(null)
                window.scrollTo({ top: 0, behavior: 'smooth' })
              }}
            >
              <div className="flex items-start">
                <span className="text-2xl mr-3 group-hover:text-red-500 transition-colors">✗</span>
                <div>
                  <p className="font-semibold text-gray-800 dark:text-gray-200 mb-2 group-hover:text-red-600 dark:group-hover:text-red-400">Hallucinated Answer Example</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    An answer that sounds plausible but is not in the document
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default AppPage
