import { useScrollReveal } from '../../hooks/useScrollReveal';

function HowItWorksSection() {
  const { ref, isVisible } = useScrollReveal({ threshold: 0.1 });

  return (
    <section id="working" className="pt-24 pb-16 px-4 bg-white dark:bg-[#0a0a0a] transition-colors duration-300 relative border-t border-gray-100 dark:border-gray-800/50 overflow-hidden">
      
      <div 
        ref={ref}
        className={`container mx-auto max-w-7xl relative z-10 transition-all duration-1000 transform ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-20'}`}
      >
        <div className="text-center mb-20 max-w-3xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-extrabold text-gray-900 dark:text-white mb-6 tracking-tight">
            The Architecture of <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 animate-text-gradient bg-[length:200%_auto]">Truth</span>
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            A symphony of internal state projection and semantic validation. Here is how our dual engine architecture guarantees uncompromising accuracy.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 relative">
          
          {/* Decorative join line for desktop */}
          <div className="hidden lg:block absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-1 bg-gradient-to-r from-purple-500/50 to-blue-500/50 blur-[2px]"></div>

          {/* Whitebox Pipeline */}
          <div className="group relative bg-white dark:bg-[#111111] p-10 rounded-3xl border border-purple-100 hover:border-purple-300 dark:border-purple-900/30 dark:hover:border-purple-500/50 transition-all duration-500 shadow-sm hover:shadow-[0_20px_50px_-5px_rgba(168,85,247,0.25)] transform hover:-translate-y-4 cursor-default">
            <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-30 group-hover:rotate-12 transition-all duration-500">
              <span className="text-8xl">🔬</span>
            </div>
            
            <div className="inline-flex items-center mb-8 bg-purple-50 dark:bg-purple-900/20 rounded-full px-4 py-1.5 border border-purple-100 dark:border-purple-800/50 hover:bg-purple-100 dark:hover:bg-purple-800/40 transition-colors duration-300">
              <div className="w-2 h-2 rounded-full bg-purple-500 mr-2 animate-pulse"></div>
              <span className="text-sm font-bold text-purple-700 dark:text-purple-300 uppercase tracking-wider">Engine 01 : Whitebox</span>
            </div>

            <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 group-hover:text-purple-500 dark:group-hover:text-purple-400 transition-colors duration-300">Internal State Tracking</h3>
            
            <div className="space-y-6 text-gray-600 dark:text-gray-400 relative border-l-2 border-purple-100 dark:border-purple-900/30 ml-3 pl-6">
              {[
                { title: 'Vector Retrieval', desc: 'Scan document via semantic search.' },
                { title: 'Model Injection', desc: 'Feed context & query into Gemma.' },
                { title: 'State Capture', desc: 'Isolate 2304 dim internal hidden states.' },
                { title: 'Dim Reduction', desc: 'Apply rapid PCA to isolate signal.' },
                { title: 'SVM Classification', desc: 'Mathematically pinpoint confidence.' },
              ].map((step, i) => (
                <div key={i} className="relative">
                  <div className="absolute -left-[31px] top-1.5 w-3 h-3 rounded-full border-2 border-purple-500 bg-white dark:bg-[#111111] group-hover:scale-125 transition-transform duration-300"></div>
                  <p className="text-lg"><span className="font-semibold text-gray-900 dark:text-gray-200">{step.title}:</span> {step.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Blackbox Pipeline */}
          <div className="group relative bg-white dark:bg-[#111111] p-10 rounded-3xl border border-blue-100 hover:border-blue-300 dark:border-blue-900/30 dark:hover:border-blue-500/50 transition-all duration-500 shadow-sm hover:shadow-[0_20px_50px_-5px_rgba(59,130,246,0.25)] mt-8 lg:mt-16 transform hover:-translate-y-4 cursor-default">
            <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-30 group-hover:-rotate-12 transition-all duration-500">
              <span className="text-8xl">🔍</span>
            </div>

            <div className="inline-flex items-center mb-8 bg-blue-50 dark:bg-blue-900/20 rounded-full px-4 py-1.5 border border-blue-100 dark:border-blue-800/50 hover:bg-blue-100 dark:hover:bg-blue-800/40 transition-colors duration-300">
              <div className="w-2 h-2 rounded-full bg-blue-500 mr-2 animate-pulse"></div>
              <span className="text-sm font-bold text-blue-700 dark:text-blue-300 uppercase tracking-wider">Engine 02 : Blackbox</span>
            </div>

            <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 group-hover:text-blue-500 dark:group-hover:text-blue-400 transition-colors duration-300">Semantic Validation</h3>
            
            <div className="space-y-6 text-gray-600 dark:text-gray-400 relative border-l-2 border-blue-100 dark:border-blue-900/30 ml-3 pl-6">
              {[
                { title: 'Intelligent Chunking', desc: 'Slice raw PDFs into localized contexts.' },
                { title: 'FAISS Indexing', desc: 'Construct sub ms exact semantic indexes.' },
                { title: 'Top K Matching', desc: 'Identify critical passage relationships.' },
                { title: 'DeBERTa NLI', desc: 'Execute natural language inference logic.' },
                { title: 'Truth Verdict', desc: 'Determine Entailment vs Contradiction.' },
              ].map((step, i) => (
                <div key={i} className="relative">
                  <div className="absolute -left-[31px] top-1.5 w-3 h-3 rounded-full border-2 border-blue-500 bg-white dark:bg-[#111111] group-hover:scale-125 transition-transform duration-300"></div>
                  <p className="text-lg"><span className="font-semibold text-gray-900 dark:text-gray-200">{step.title}:</span> {step.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-20 text-center relative z-20">
          <div className="inline-flex flex-col items-center p-6 rounded-2xl bg-gradient-to-r from-purple-50/50 to-blue-50/50 dark:from-purple-900/10 dark:to-blue-900/10 border border-gray-200 dark:border-gray-800 backdrop-blur-md">
             <div className="flex items-center gap-3 mb-2">
               <span className="px-3 py-1 rounded bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 font-mono text-sm font-bold">IF (E1 & E2 == AGREE)</span>
             </div>
             <p className="text-xl font-medium text-gray-800 dark:text-gray-200">
               Ultimate zero trust hallucination verdict.
             </p>
          </div>
        </div>

      </div>
    </section>
  )
}

export default HowItWorksSection
