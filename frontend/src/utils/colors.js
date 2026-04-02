/**
 * Utility functions for verdict colors, icons, and labels
 */

export const getVerdictColor = (verdict) => {
  const colorMap = {
    // Whitebox labels
    CORRECT: 'green',
    HALLUCINATED: 'red',
    // Blackbox labels
    GROUNDED: 'green',
    UNCERTAIN: 'yellow',
    HALLUCINATION: 'red',
  }
  return colorMap[verdict] || 'gray'
}

export const getVerdictIcon = (verdict) => {
  const iconMap = {
    CORRECT: '✓',
    HALLUCINATED: '✗',
    GROUNDED: '✓',
    UNCERTAIN: '⚠',
    HALLUCINATION: '✗',
  }
  return iconMap[verdict] || '?'
}

export const getVerdictClasses = (verdict) => {
  const color = getVerdictColor(verdict)
  const classMap = {
    green: 'bg-green-50/80 dark:bg-green-900/20 border-green-500 text-green-900 dark:text-green-100',
    red: 'bg-red-50/80 dark:bg-red-900/20 border-red-500 text-red-900 dark:text-red-100',
    yellow: 'bg-yellow-50/80 dark:bg-yellow-900/20 border-yellow-500 text-yellow-900 dark:text-yellow-100',
    gray: 'bg-gray-50/80 dark:bg-gray-900/20 border-gray-500 text-gray-900 dark:text-gray-100',
  }
  return classMap[color] || classMap.gray
}

export const getBadgeClasses = (verdict) => {
  const color = getVerdictColor(verdict)
  const classMap = {
    green: 'bg-green-100 dark:bg-green-900/40 text-green-800 dark:text-green-300 border-green-300 dark:border-green-800/50',
    red: 'bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-300 border-red-300 dark:border-red-800/50',
    yellow: 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-800 dark:text-yellow-300 border-yellow-300 dark:border-yellow-800/50',
    gray: 'bg-gray-100 dark:bg-gray-900/40 text-gray-800 dark:text-gray-300 border-gray-300 dark:border-gray-800/50',
  }
  return classMap[color] || classMap.gray
}
