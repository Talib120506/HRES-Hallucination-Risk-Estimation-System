import { useEffect, useRef, useState } from 'react';

export const useScrollReveal = (options = { threshold: 0.1, rootMargin: '0px' }) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      // Update state whenever it enters or leaves the viewport.
      // This ensures animations trigger multiple times on scroll.
      setIsVisible(entry.isIntersecting);
    }, options);

    const currentRef = ref.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (currentRef) {
        observer.unobserve(currentRef);
      }
    };
  }, [options.rootMargin, options.threshold]);

  return { ref, isVisible };
};
