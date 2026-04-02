import { Link } from 'react-router-dom'
import { useParallax } from '../../hooks/useParallax'

function HeroSection() {
  const bgOffset = useParallax(0.4);
  const elementsOffset = useParallax(0.15);

  return (
    <section className="relative overflow-hidden bg-white dark:bg-[#0a0a0a] transition-colors duration-300 min-h-[90vh] flex items-center pt-20">
      {/* Background Decorative Blobs - Fixed Parallax Background */}
      <div 
        className="absolute inset-0 z-0 pointer-events-none"
        style={{ transform: `translateY(${bgOffset}px)` }}
      >
        <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-300 dark:bg-purple-900 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob"></div>
        <div className="absolute top-0 -right-4 w-72 h-72 bg-blue-300 dark:bg-blue-900 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-indigo-300 dark:bg-indigo-900 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-4000"></div>
      </div>

      <div 
        className="container relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 z-10"
        style={{ transform: `translateY(${elementsOffset}px)` }}
      >
        <div className="text-center max-w-4xl mx-auto opacity-0 animate-[fadeInUp_0.8s_ease-out_forwards]">
          {/* Version / Release pill */}
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-purple-200 dark:border-purple-800/50 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 text-sm font-medium mb-8 hover:scale-105 hover:bg-purple-100 dark:hover:bg-purple-800/40 transition-all duration-300 cursor-default shadow-sm hover:shadow-purple-500/20">
            <span className="flex h-2 w-2 rounded-full bg-purple-600 dark:bg-purple-400 animate-pulse"></span>
            Reimagining AI Trust
          </div>

<h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8 text-gray-900 dark:text-white transform transition-transform duration-500 hover:scale-[1.01] leading-tight">
            Trust your AI Models with <br className="hidden md:block"/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#7c3aed] via-[#f43f5e] to-[#3b82f6] animate-text-gradient hover:from-[#9333ea] hover:to-[#2563eb] transition-colors duration-300 bg-[length:200%_auto] pb-2 inline-block">      
              Clinical Precision.
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl mb-12 text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
            HRES uses a revolutionary <span className="text-gray-900 dark:text-white font-semibold">Dual-Pipeline System</span> (Whitebox & Blackbox) to mathematically detect, estimate, and eliminate hallucinations in large language models.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              to="/app"
              className="w-full sm:w-auto px-8 py-4 rounded-xl text-lg font-semibold text-white bg-gray-900 dark:bg-white dark:text-gray-900 hover:bg-gray-800 dark:hover:bg-gray-100 shadow-[0_0_40px_-10px_rgba(124,58,237,0.5)] transition-all duration-300 hover:scale-105 hover:shadow-[0_0_60px_-15px_rgba(124,58,237,0.7)]"
            >
              Start Verifying →
            </Link>
          </div>

          {/* Miniature UI Mockup/Dashboard Preview */}
          <div className="mt-20 relative mx-auto w-full max-w-5xl rounded-2xl border border-gray-200/50 dark:border-gray-800 bg-white/50 dark:bg-[#111111]/80 backdrop-blur-xl shadow-2xl p-2 sm:p-4 transform hover:scale-[1.02] hover:-translate-y-2 transition-all duration-500 hover:shadow-[0_20px_60px_-15px_rgba(124,58,237,0.3)] group cursor-default">
            <div className="rounded-xl overflow-hidden border border-gray-100 dark:border-gray-800/50 bg-gray-50 dark:bg-[#0a0a0a] min-h-[300px] flex flex-col relative z-20 group-hover:border-purple-300/50 dark:group-hover:border-purple-500/30 transition-colors duration-500">
              {/* Fake Mac Header */}
              <div className="h-10 border-b border-gray-200 dark:border-gray-800/50 flex items-center px-4 gap-2">
                <div className="w-3 h-3 rounded-full bg-red-400 hover:bg-red-500 transition-colors"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-400 hover:bg-yellow-500 transition-colors"></div>
                <div className="w-3 h-3 rounded-full bg-green-400 hover:bg-green-500 transition-colors"></div>
              </div>
              <div className="flex-1 p-6 flex flex-col items-center justify-center text-gray-400 dark:text-gray-600">
                {/* Abstract Data Viz */}
                <div className="grid grid-cols-3 gap-6 w-full max-w-3xl mb-8">
                  <div className="h-24 rounded-lg bg-gradient-to-br from-purple-100 to-white dark:from-purple-900/20 dark:to-transparent border border-purple-200 dark:border-purple-800/30 p-4 transform hover:scale-105 hover:-translate-y-1 transition-all duration-300">
                    <div className="h-3 w-1/2 bg-purple-200 dark:bg-purple-800/50 rounded mb-4"></div>
                    <div className="text-3xl font-bold text-purple-700 dark:text-purple-400">98.4%</div>
                  </div>
                  <div className="h-24 rounded-lg bg-gradient-to-br from-blue-100 to-white dark:from-blue-900/20 dark:to-transparent border border-blue-200 dark:border-blue-800/30 p-4 transform hover:scale-105 hover:-translate-y-1 transition-all duration-300">
                    <div className="h-3 w-1/2 bg-blue-200 dark:bg-blue-800/50 rounded mb-4"></div>
                    <div className="text-3xl font-bold text-blue-700 dark:text-blue-400">3.2ms</div>
                  </div>
                  <div className="h-24 rounded-lg bg-gray-100 dark:bg-gray-800/30 border border-gray-200 dark:border-gray-800/50 p-4 transform hover:scale-105 hover:-translate-y-1 transition-all duration-300">
                    <div className="h-3 w-2/3 bg-gray-200 dark:bg-gray-700 rounded mb-4"></div>
                    <div className="text-xl font-medium text-gray-500 font-mono group-hover:text-green-500 dark:group-hover:text-green-400 transition-colors duration-300">SECURE</div>
                  </div>
                </div>
                <div className="h-40 w-full max-w-3xl rounded-lg bg-gray-100 dark:bg-gradient-to-t dark:from-gray-800/40 dark:to-transparent border border-gray-200 dark:border-gray-800 flex items-end overflow-hidden group-hover:border-purple-200 dark:group-hover:border-purple-800/50 transition-colors duration-500">
                   <div className="w-full h-1/2 bg-gradient-to-t from-purple-500/20 to-transparent transform translate-y-full group-hover:translate-y-0 transition-transform duration-1000 ease-in-out"></div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  )
}

export default HeroSection
