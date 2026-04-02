import React, { useState } from 'react';
import { useScrollReveal } from '../../hooks/useScrollReveal';

const FAQItem = ({ question, answer }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="border-b border-gray-200 dark:border-gray-800 group">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full py-6 flex justify-between items-center focus:outline-none text-left transform transition-transform duration-300 hover:translate-x-2"
      >
        <span className="text-lg font-medium text-gray-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors duration-300">
          {question}
        </span>
        <span className="ml-6 flex-shrink-0">
          <svg
            className={`w-6 h-6 text-gray-500 group-hover:text-purple-500 transform transition-all duration-300 ${
              isOpen ? 'rotate-180 text-purple-600 dark:text-purple-400' : ''
            }`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </span>
      </button>
      <div
        className={`overflow-hidden transition-all duration-300 ease-in-out ${
          isOpen ? 'max-h-96 pb-6 opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          {answer}
        </p>
      </div>
    </div>
  );
};

const FAQSection = () => {
  const { ref, isVisible } = useScrollReveal({ threshold: 0.1 });

  const faqs = [
    {
      question: "What is HRES?",
      answer: "HRES (Hallucination Risk Estimation System) is a tool designed to analyze AI-generated outputs and assess their risk of containing hallucinations—information that is factually incorrect or unsupported by the given context."
    },
    {
      question: "How does the evaluation process work?",
      answer: "We use a multi-step process that compares the generated text against verified contexts using natural language inference (NLI) techniques. The system breaks down the text, checks factual alignment, and provides a confidence score indicating the likelihood of hallucination."
    },
    {
      question: "Can I use HRES with my own domain-specific data?",
      answer: "Yes, HRES is designed to be adaptable. You can index your own domain-specific documents, which the system will use as the ground truth context when evaluating new text generations."
    },
    {
      question: "What formats do you support for context documents?",
      answer: "Currently, we support PDF and plain text documents. We are actively working on expanding our ingestion pipelines to process Word documents, internal wikis, and direct database integrations."
    }
  ];

  return (
    <section id="faq" className="pt-8 pb-24 bg-white dark:bg-[#0a0a0a] transition-colors duration-300">
      <div 
        ref={ref}
        className={`container mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 transition-all duration-1000 transform ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-16 scale-95'}`}
      >
        <div className="text-center mb-16">
          <h2 className="text-sm font-bold tracking-widest text-[#7c3aed] uppercase mb-3">FAQ</h2>
          <h3 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-900 via-gray-600 to-gray-900 dark:from-white dark:via-gray-400 dark:to-white animate-text-gradient bg-[length:200%_auto] tracking-tight mb-4">
            Frequently Asked Questions
          </h3>
          <p className="mt-4 text-xl text-gray-600 dark:text-gray-400 font-light">
            Everything you need to know about our project and how it works.
          </p>
        </div>

        <div className="bg-gray-50 dark:bg-[#111111] p-8 md:p-12 rounded-3xl border border-gray-100 dark:border-gray-800 shadow-sm dark:shadow-none">
          <div className="divide-y divide-gray-200 dark:divide-gray-800">
            {faqs.map((faq, index) => (
              <FAQItem key={index} question={faq.question} answer={faq.answer} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default FAQSection;
