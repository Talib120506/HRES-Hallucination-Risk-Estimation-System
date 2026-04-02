import React, { useState } from 'react';
import { useScrollReveal } from '../../hooks/useScrollReveal';

const ContactSection = () => {
  const { ref, isVisible } = useScrollReveal({ threshold: 0.15 });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);

  // NOTE: Replace this with your actual Google Apps Script Web App URL
  const GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbylUI4mC6Kmm24M8tRK3fBrUrA4VpCQk4WhMJFMov7PUOqMpsmUgBz4hiFsOQdEzBMHdw/exec";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus(null);
    
    const form = e.target;
    // To support no-cors, we can use FormData directly.
    const formData = new FormData(form);
    // Convert to URLSearchParams to guarantee compatibility with Google Apps Script doPost parameter reading
    const urlEncoded = new URLSearchParams(formData).toString();

    try {
      await fetch(GOOGLE_SCRIPT_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: urlEncoded,
        mode: 'no-cors' // Use no-cors to prevent CORS issues when submitting form data to Google Apps Script
      });
      
      // With no-cors, the response is opaque, so we assume success if no error is thrown
      setSubmitStatus('success');
      form.reset();
    } catch (error) {
      console.error('Error submitting form:', error);
      setSubmitStatus('error');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section id="contact" className="py-24 bg-gray-50 dark:bg-[#0a0a0a] transition-colors duration-300 border-t border-gray-200 dark:border-gray-800/50 overflow-hidden">
      <div 
        ref={ref}
        className={`container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 transition-all duration-1000 transform ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-24'}`}
      >
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-16">
            <h3 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 via-pink-600 to-purple-600 dark:from-purple-400 dark:via-pink-400 dark:to-purple-400 animate-text-gradient bg-[length:200%_auto] tracking-tight mb-4">
              Get in Touch
            </h3>
          </div>

          <div className="bg-white dark:bg-[#111111] border border-gray-100 dark:border-gray-800 p-8 md:p-12 rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:shadow-none transform transition-all duration-700 hover:-translate-y-2 hover:shadow-[0_20px_50px_-15px_rgba(124,58,237,0.2)] hover:border-purple-300/50 dark:hover:border-purple-500/30">
            <form className="space-y-6 group" onSubmit={handleSubmit}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="transform transition-transform duration-300 focus-within:-translate-y-1">
                  <label htmlFor="name" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 transition-colors group-focus-within:text-purple-600 dark:group-focus-within:text-purple-400">
                    Full Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    id="name"
                    required
                    disabled={isSubmitting}
                    className="w-full bg-gray-50 dark:bg-[#0a0a0a] border border-gray-200 dark:border-gray-800 rounded-xl px-4 py-3 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all shadow-sm hover:shadow-md"
                    placeholder="Jane Doe"
                  />
                </div>
                <div className="transform transition-transform duration-300 focus-within:-translate-y-1">
                  <label htmlFor="email" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 transition-colors group-focus-within:text-purple-600 dark:group-focus-within:text-purple-400">
                    Work Email
                  </label>
                  <input
                    type="email"
                    name="email"
                    id="email"
                    required
                    disabled={isSubmitting}
                    className="w-full bg-gray-50 dark:bg-[#0a0a0a] border border-gray-200 dark:border-gray-800 rounded-xl px-4 py-3 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all shadow-sm hover:shadow-md"
                    placeholder="jane@company.com"
                  />
                </div>
              </div>
              <div>
                <label htmlFor="message" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  How can we help?
                </label>
                <textarea
                  id="message"
                  name="message"
                  rows={4}
                  required
                  disabled={isSubmitting}
                  className="w-full bg-gray-50 dark:bg-[#0a0a0a] border border-gray-200 dark:border-gray-800 rounded-xl px-4 py-3 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all shadow-sm hover:shadow-md resize-none"
                  placeholder="Tell us about your use case..."
                />
              </div>
              
              {submitStatus === 'success' && (
                <div className="p-4 bg-green-50 dark:bg-green-900/20 shadow-sm border border-green-500/30 text-green-700 dark:text-green-400 rounded-xl text-center font-medium animate-fade-in-up">
                  Thanks for reaching out! We'll get back to you shortly.
                </div>
              )}
              {submitStatus === 'error' && (
                <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-500/30 text-red-700 dark:text-red-400 rounded-xl text-center font-medium animate-fade-in-up">
                  Oops! Something went wrong. Please try again.
                </div>
              )}

              <div className="pt-4 transform transition-transform duration-300">
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full flex justify-center py-4 px-4 rounded-xl text-lg font-bold text-white bg-gray-900 hover:bg-black dark:bg-white dark:text-gray-900 dark:hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-900 dark:focus:ring-white transition-all transform hover:scale-[1.03] hover:-translate-y-1 hover:shadow-[0_15px_40px_-10px_rgba(124,58,237,0.5)] active:scale-95 disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {isSubmitting ? (
                    <span className="flex items-center gap-2">
                       <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                         <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                         <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                       </svg>
                       Sending...
                    </span>
                  ) : (
                    "Send Message"
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ContactSection;
