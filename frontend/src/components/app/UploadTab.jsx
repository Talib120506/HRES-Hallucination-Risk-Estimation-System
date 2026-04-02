import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

function UploadTab({ formData, setFormData, onSubmit, errors }) {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles && acceptedFiles.length > 0) {
        const file = acceptedFiles[0]
        setFormData({ ...formData, file })
      }
    },
    [formData, setFormData]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
  })

  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Upload PDF
        </label>
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-500 transform hover:-translate-y-1 hover:shadow-[0_10px_30px_-10px_rgba(124,58,237,0.3)] ${
            isDragActive
              ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20 scale-[1.02] shadow-[0_0_30px_-5px_rgba(124,58,237,0.4)]'
              : 'border-gray-300 dark:border-gray-700 hover:border-purple-400 dark:hover:border-purple-500 hover:bg-gray-50/50 dark:hover:bg-purple-900/10'
          }`}
        >
          <input {...getInputProps()} />
          <div className="text-gray-600 dark:text-gray-400 flex flex-col items-center">
            {formData.file ? (
              <>
                <svg className="w-12 h-12 text-purple-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                <p className="text-lg font-semibold text-purple-700 dark:text-purple-400">
                  {formData.file.name}
                </p>
                <p className="text-sm font-medium mt-1">
                  {(formData.file.size / 1024 / 1024).toFixed(2)} MB
                </p>
                <p className="text-xs opacity-70 mt-2">
                  Click or drag to replace
                </p>
              </>
            ) : isDragActive ? (
              <>
                <svg className="w-12 h-12 text-purple-500 mb-3 animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                <p className="text-lg font-medium text-purple-600 dark:text-purple-400">Drop your PDF here...</p>
              </>
            ) : (
              <>
                <svg className="w-12 h-12 opacity-50 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
                  Drag & drop a PDF here, or <span className="text-purple-600 dark:text-purple-400">click to select</span>
                </p>
                <p className="text-sm mt-2">PDF files only, max 50MB</p>
              </>
            )}
          </div>
        </div>
        {errors.pdf && (
          <p className="mt-2 text-sm text-red-600 dark:text-red-400 flex items-center gap-1"><span className="text-lg leading-none">&times;</span> {errors.pdf}</p>
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
          <p className="mt-2 text-sm text-red-600 dark:text-red-400 flex items-center gap-1"><span className="text-lg leading-none">&times;</span> {errors.question}</p>
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
          <p className="mt-2 text-sm text-red-600 dark:text-red-400 flex items-center gap-1"><span className="text-lg leading-none">&times;</span> {errors.answer}</p>
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

export default UploadTab
