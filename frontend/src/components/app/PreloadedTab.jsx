import { useState, useEffect } from 'react'
import { fetchPreloadedPDFs } from '../../services/api'

function PreloadedTab({ formData, setFormData, onSubmit, errors }) {
  const [pdfs, setPdfs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const loadPDFs = async () => {
      try {
        setLoading(true)
        const data = await fetchPreloadedPDFs()
        setPdfs(data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadPDFs()
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Select Document
        </label>
        {loading ? (
          <div className="text-gray-500 dark:text-gray-400">Loading PDFs...</div>
        ) : error ? (
          <div className="text-red-600 dark:text-red-400">{error}</div>
        ) : (
          <select
            className="w-full bg-white dark:bg-[#0a0a0a] border border-gray-200 dark:border-gray-700/50 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 outline-none text-gray-900 dark:text-white shadow-sm hover:shadow-md focus:shadow-lg focus:shadow-purple-500/20 transform focus:-translate-y-0.5 cursor-pointer"
            value={formData.preloaded_pdf || ''}
            onChange={(e) =>
              setFormData({ ...formData, preloaded_pdf: e.target.value })
            }
          >
            <option value="">-- Select a PDF --</option>
            {pdfs.map((pdf) => (
              <option key={pdf.filename} value={pdf.filename}>
                {pdf.display_name}
              </option>
            ))}
          </select>
        )}
        {errors.pdf && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.pdf}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Question
        </label>
        <input
          type="text"
          className="w-full bg-white dark:bg-[#0a0a0a] border border-gray-200 dark:border-gray-700/50 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 outline-none text-gray-900 dark:text-white shadow-sm hover:shadow-md focus:shadow-lg focus:shadow-purple-500/20 transform focus:-translate-y-0.5 placeholder-gray-400 dark:placeholder-gray-500"
          placeholder="Enter your question about the document..."
          value={formData.question}
          onChange={(e) =>
            setFormData({ ...formData, question: e.target.value })
          }
        />
        {errors.question && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.question}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Answer to Verify
        </label>
        <textarea
          className="w-full bg-white dark:bg-[#0a0a0a] border border-gray-200 dark:border-gray-700/50 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 outline-none text-gray-900 dark:text-white shadow-sm hover:shadow-md focus:shadow-lg focus:shadow-purple-500/20 transform focus:-translate-y-0.5 placeholder-gray-400 dark:placeholder-gray-500 min-h-[120px] resize-none"
          rows="4"
          placeholder="Enter the answer you want to verify for hallucinations..."
          value={formData.answer}
          onChange={(e) => setFormData({ ...formData, answer: e.target.value })}
        />
        {errors.answer && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.answer}</p>
        )}
      </div>

      <button
        onClick={onSubmit}
        className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-4 rounded-xl font-bold text-lg hover:from-purple-500 hover:to-blue-500 transition-all duration-300 transform hover:-translate-y-1 shadow-[0_0_20px_-5px_rgba(124,58,237,0.5)] hover:shadow-[0_0_30px_-5px_rgba(124,58,237,0.7)]"
      >
        Analyze Answer
      </button>
    </div>
  )
}

export default PreloadedTab
