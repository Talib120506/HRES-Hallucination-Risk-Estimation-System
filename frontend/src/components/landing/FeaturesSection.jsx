import { useScrollReveal } from '../../hooks/useScrollReveal';
import { useParallax } from '../../hooks/useParallax';

function FeaturesSection() {
  const { ref, isVisible } = useScrollReveal({ threshold: 0.15 });
  const bgOffset = useParallax(0.18);

  const features = [
    {
      icon: '🔬',
      title: 'Dual-Pipeline Setup',
      description: 'Our proprietary approach merges internal model analysis with natural language inference, catching nuance that basic prompt-checking misses.',
      colSpan: 'lg:col-span-2'
    },
    {
      icon: '⚡',
      title: 'Real-Time Edge',
      description: 'Get immediate feedback on prompt safety and hallucination risks with blazing mathematical efficiency.',
      colSpan: 'lg:col-span-1'
    },
    {
      icon: '🔒',
      title: 'Local Privacy',
      description: 'No external APIs. Your data never leaves your environment. Perfect for enterprise & legal data.',
      colSpan: 'lg:col-span-1'
    },
    {
      icon: '🎯',
      title: 'Clinical Accuracy',
      description: 'Trained rigorously against adversarial sets to discern subtle logical missteps from genuine creativity.',
      colSpan: 'lg:col-span-2'
    },
  ]

  return (
    <section id="features" className="py-24 px-4 bg-gray-50 dark:bg-[#0a0a0a] transition-colors duration-300 relative border-t border-gray-200 dark:border-gray-800/50">
      {/* Background Glow */}
      <div 
        className="absolute top-40 left-1/2 -translate-x-1/2 w-3/4 h-64 bg-blue-500/10 dark:bg-blue-600/10 filter blur-[100px] rounded-full pointer-events-none"
        style={{ transform: `translate(-50%, ${bgOffset}px)` }}
      ></div>

      <div 
        ref={ref}
        className={`container mx-auto max-w-6xl relative z-10 transition-all duration-1000 transform ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-16 scale-95'}`}
      >
        <div className="text-center mb-16 max-w-2xl mx-auto">
          <h2 className="text-sm font-bold tracking-widest text-[#7c3aed] uppercase mb-3">Enterprise Features</h2>
          <h3 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-900 via-purple-800 to-gray-900 dark:from-white dark:via-purple-300 dark:to-white animate-text-gradient bg-[length:200%_auto] pb-2 leading-tight mb-4">
            Security at the speed of thought.
          </h3>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Engineered from the ground up for maximum precision without compromising performance.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 auto-rows-fr">
          {features.map((feature, index) => (
            <div
              key={index}
              className={`group flex flex-col justify-between overflow-hidden bg-white dark:bg-[#111111] p-8 rounded-3xl border border-gray-200 hover:border-purple-300 dark:border-gray-800 dark:hover:border-purple-500/50 transition-all duration-500 hover:shadow-[0_20px_40px_rgba(124,58,237,0.15)] transform hover:-translate-y-3 cursor-pointer ${feature.colSpan || ''}`}
            >
              <div>
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gray-50 dark:bg-gray-800 mb-6 text-2xl group-hover:scale-110 group-hover:bg-purple-50 dark:group-hover:bg-purple-900/30 transition-all duration-500">
                  {feature.icon}
                </div>
                <h4 className="text-2xl font-bold mb-3 text-gray-900 dark:text-gray-100 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-purple-500 group-hover:to-blue-500 transition-colors duration-300">
                  {feature.title}
                </h4>
                <p className="text-gray-600 dark:text-gray-400 leading-relaxed text-lg">
                  {feature.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default FeaturesSection
