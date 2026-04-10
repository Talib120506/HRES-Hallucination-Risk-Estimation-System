import Navbar from '../components/landing/Navbar'
import HeroSection from '../components/landing/HeroSection'
import AboutSection from '../components/landing/AboutSection'
import FeaturesSection from '../components/landing/FeaturesSection'
import HowItWorksSection from '../components/landing/HowItWorksSection'
import FAQSection from '../components/landing/FAQSection'
import ContactSection from '../components/landing/ContactSection'
import Footer from '../components/landing/Footer'

function LandingPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-[#0a0a0a] transition-colors duration-300">
      <Navbar />
      {/* Add padding top to account for fixed navbar */}
      <div className="pt-20">
        <HeroSection />
        <AboutSection />
        <FeaturesSection />
        <HowItWorksSection />
        <FAQSection />
        <ContactSection />
        <Footer />
      </div>
    </div>
  )
}

export default LandingPage
