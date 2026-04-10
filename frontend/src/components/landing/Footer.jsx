import React from 'react';

const Footer = () => {
  const developers = [
    {
      name: "Krrish Sarbhai",
      github: "https://github.com/krrishsarbhai",
      linkedin: "https://www.linkedin.com/in/krrish-sarbhai-82a9b134b?utm_source=share_via&utm_content=profile&utm_medium=member_android"
    },
    {
      name: "Talib Jiruwala",
      github: "https://github.com/Talib120506",
      linkedin: "https://www.linkedin.com/in/talib-kosar-jiruwala-376a03256?utm_source=share_via&utm_content=profile&utm_medium=member_android"
    },
    {
      name: "Prachi Sangaonkar",
      github: "https://github.com/PrachiSangaonkar",
      linkedin: "https://www.linkedin.com/in/prachi-sangaonkar-353ba12bb/"
    }
  ];

  return (
    <footer className="relative bg-white dark:bg-[#030303] border-t border-gray-200 dark:border-gray-800/60 text-gray-800 dark:text-gray-200 z-20 overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-[100px] -z-10 pointer-events-none"></div>
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[100px] -z-10 pointer-events-none"></div>

      <div className="max-w-7xl mx-auto px-6 lg:px-8 pt-10 pb-4 md:pt-16 md:pb-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-8 items-center lg:items-start">
          
          {/* Left Side: Brand and Contribute */}
          <div className="flex flex-col items-center lg:justify-center gap-8 text-center">
            <div>
              <h2 className="text-6xl md:text-8xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 drop-shadow-sm">
                HRES.
              </h2>
              <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-400 mt-4 font-medium max-w-md leading-relaxed">
                Hallucination Risk Estimation System
              </p>
            </div>
            
            <a 
              href="https://github.com/Talib120506/HRES-Hallucination-Risk-Estimation-System" 
              target="_blank" 
              rel="noopener noreferrer"
              className="mt-6 group relative inline-flex items-center justify-center gap-3 px-8 py-4 bg-gray-900 dark:bg-white text-white dark:text-gray-900 text-lg font-bold rounded-full overflow-hidden transition-all duration-300 hover:scale-105 active:scale-95 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.3)] dark:shadow-[0_10px_40px_-10px_rgba(255,255,255,0.2)]"
            >
              <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-blue-600 to-purple-600 opacity-0 group-hover:opacity-20 dark:group-hover:opacity-10 transition-opacity duration-300"></div>
              <span className="relative z-10">Wanna Contribute?</span>
              <svg className="w-6 h-6 relative z-10 group-hover:rotate-12 transition-transform duration-300" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
              </svg>
            </a>
          </div>

          {/* Right Side: Developers */}
          <div className="flex flex-col items-center lg:items-end w-full">
            <div className="w-full sm:w-[400px] lg:w-auto px-8 py-6 rounded-3xl bg-gray-50/50 dark:bg-white/[0.02] border border-gray-100 dark:border-white/5 backdrop-blur-sm">
              <h3 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white mb-6 flex items-center lg:justify-end gap-4">
                <span className="w-12 h-1 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full hidden lg:block"></span>
                The Developers
                <span className="w-12 h-1 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full lg:hidden block"></span>
              </h3>
              
              <ul className="space-y-4 lg:min-w-[320px]">
                {developers.map((dev, idx) => (
                  <li key={idx} className="flex items-center justify-between gap-8 group">
                    <span className="text-xl text-gray-800 dark:text-gray-200 font-bold group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                      {dev.name}
                    </span>
                    <div className="flex items-center gap-5">
                      <a 
                        href={dev.github} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-gray-400 hover:text-gray-900 dark:hover:text-white transform hover:scale-110 transition-all duration-300" 
                        title={`${dev.name} GitHub`}
                      >
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" /></svg>
                      </a>
                      <a 
                        href={dev.linkedin} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transform hover:scale-110 transition-all duration-300" 
                        title={`${dev.name} LinkedIn`}
                      >
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path fillRule="evenodd" d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" clipRule="evenodd" /></svg>
                      </a>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          
        </div>
      </div>
    </footer>
  );
};

export default Footer;