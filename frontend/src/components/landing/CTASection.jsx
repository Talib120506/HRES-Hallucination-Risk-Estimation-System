import { Link } from 'react-router-dom'

function CTASection() {
  return (
    <section className="py-24 px-4 bg-white dark:bg-[#0a0a0a] transition-colors duration-300 relative border-t border-gray-100 dark:border-gray-800/50">
      <div className="container mx-auto max-w-5xl text-center relative z-10">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-blue-500 blur-[100px] opacity-10 dark:opacity-20 pointer-events-none"></div>
        
        <div className="relative bg-black dark:bg-[#111] overflow-hidden rounded-[3rem] px-8 py-20 border border-gray-800 shadow-2xl">
          <div className="absolute inset-0 opacity-30 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-purple-900 via-transparent to-transparent"></div>
          
          <h2 className="text-4xl md:text-6xl font-extrabold mb-8 text-white tracking-tight">
            Ready to deploy with confidence?
          </h2>
          <p className="text-xl mb-12 text-gray-300 max-w-2xl mx-auto font-light">
            Stop guessing. Start knowing. Try HRES with your own PDFs or our preloaded enterprise examples today.
          </p>
          
          <Link
            to="/app"
            className="inline-flex items-center justify-center px-10 py-5 rounded-full text-lg font-bold text-white bg-gradient-to-r from-[#7c3aed] to-[#3b82f6] hover:scale-105 hover:shadow-[0_0_40px_-10px_rgba(124,58,237,0.8)] transition-all duration-300"
          >
            Launch Web App
            <svg className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>

          <div className="mt-16 pt-8 border-t border-gray-800 flex flex-col items-center justify-center space-y-2">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400 font-bold tracking-widest text-sm uppercase">
              HRES
            </span>
            <p className="text-gray-500 text-sm">
              Hallucination Risk Estimation System &copy; {new Date().getFullYear()}
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default CTASection
