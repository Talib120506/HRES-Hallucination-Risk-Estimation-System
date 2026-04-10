import { useScrollReveal } from '../../hooks/useScrollReveal';
import { useParallax } from '../../hooks/useParallax';

function AboutSection() {
  const { ref, isVisible } = useScrollReveal({ threshold: 0.15 });
  const bgOffset = useParallax(0.12);

  return (
    <section id="about" className="py-24 px-4 bg-white dark:bg-[#0a0a0a] transition-colors duration-300 relative overflow-hidden">
      {/* Decorative left-edge glow with Parallax effect */}
      <div 
        className="absolute top-1/2 left-0 w-1/3 h-full bg-purple-500/10 filter blur-[120px] pointer-events-none"
        style={{ transform: `translateY(calc(-50% + ${bgOffset}px))` }}
      ></div>

      <div
        ref={ref}
        className={`container mx-auto max-w-7xl relative z-10 transition-all duration-1000 transform ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-16'}`}
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          
          <div className="order-2 lg:order-1 relative group cursor-default">
             <div className="absolute inset-0 bg-gradient-to-br from-[#7c3aed]/20 to-[#3b82f6]/20 rounded-3xl blur-2xl transform -rotate-6 scale-105 opacity-50 dark:opacity-100 group-hover:rotate-0 transition-transform duration-700 ease-out"></div>
             <div className="relative bg-white dark:bg-[#111111] border border-gray-100 dark:border-gray-800 rounded-3xl p-8 shadow-2xl transform group-hover:scale-[1.03] transition-transform duration-700 ease-out group-hover:shadow-[0_20px_60px_-15px_rgba(59,130,246,0.3)]">
                <div className="space-y-4">
                  <div className="flex justify-between items-center pb-4 border-b border-gray-100 dark:border-gray-800">
                    <span className="text-gray-500 text-sm font-mono tracking-widest group-hover:text-blue-500 transition-colors duration-500">SYSTEM STATUS</span>
                    <span className="flex items-center text-green-500 text-sm font-medium"><div className="w-2 h-2 rounded-full bg-green-500 mr-2 animate-pulse shadow-[0_0_8px_2px_rgba(34,197,94,0.5)]"></div> ONLINE</span>
                  </div>
                  <div className="space-y-2 pt-2">
                    <div className="h-4 w-3/4 rounded bg-gray-100 dark:bg-gray-800 group-hover:bg-blue-50 dark:group-hover:bg-blue-900/20 transition-colors duration-500"></div>
                    <div className="h-4 w-full rounded bg-gray-100 dark:bg-gray-800 group-hover:bg-purple-50 dark:group-hover:bg-purple-900/20 transition-colors duration-500 delay-75"></div>
                    <div className="h-4 w-5/6 rounded bg-gray-100 dark:bg-gray-800 group-hover:bg-indigo-50 dark:group-hover:bg-indigo-900/20 transition-colors duration-500 delay-150"></div>
                  </div>
                  <div className="mt-8 p-4 rounded-xl bg-red-50 dark:bg-red-900/10 border border-red-100 dark:border-red-900/30 transform group-hover:-translate-y-2 transition-transform duration-500 group-hover:shadow-lg group-hover:shadow-red-500/20">
                     <p className="text-red-600 dark:text-red-400 font-mono text-sm group-hover:animate-pulse">[!] HALLUCINATION DETECTED</p>
                  </div>
                </div>
             </div>
          </div>

          <div className="order-1 lg:order-2 space-y-8">
            <div>
              <h2 className="text-sm font-bold tracking-widest text-[#3b82f6] uppercase mb-3">About HRES</h2>
              <h3 className="text-4xl md:text-5xl font-extrabold text-gray-900 dark:text-white leading-tight">
                Don't guess if the AI is lying. <br className="hidden lg:block"/> <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-500 via-indigo-400 to-purple-500 animate-text-gradient bg-[length:200%_auto]">Know.</span>
              </h3>
            </div>
            
            <div className="space-y-6 text-xl text-gray-600 dark:text-gray-300 leading-relaxed font-light">
              <p>
                As LLMs integrate into critical workflows, their tendency to "hallucinate" plausible but false information has become the biggest barrier to enterprise adoption.
              </p>
              <p>
                HRES is an advanced detection framework that fundamentally shifts trust. Using a dual pipeline approach, we don't just ask the model if it's sure we look at the <strong className="text-gray-900 dark:text-white group-hover:text-purple-500 transition-colors">math inside its hidden layers</strong> to prove it.
              </p>
              <div className="pt-4 flex flex-col sm:flex-row gap-4 h-full">
                <div className="flex-1 p-4 rounded-2xl bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-800 hover:border-purple-300 dark:hover:border-purple-700/50 hover:bg-white dark:hover:bg-[#151515] transition-all duration-300 transform hover:-translate-y-2 hover:shadow-xl hover:shadow-purple-500/10 cursor-default group">
                  <h4 className="font-bold text-gray-900 dark:text-white mb-1 tracking-tight group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">Whitebox</h4>
                  <p className="text-sm">Analyzes layer representations for statistical uncertainty.</p>
                </div>
                <div className="flex-1 p-4 rounded-2xl bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-800 hover:border-blue-300 dark:hover:border-blue-700/50 hover:bg-white dark:hover:bg-[#151515] transition-all duration-300 transform hover:-translate-y-2 hover:shadow-xl hover:shadow-blue-500/10 cursor-default group">
                  <h4 className="font-bold text-gray-900 dark:text-white mb-1 tracking-tight group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">Blackbox</h4>
                  <p className="text-sm">Cross references claims via semantic search & NLI.</p>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  )
}

export default AboutSection
