/**
 * Loading Spinner Component
 */

function LoadingSpinner({ size = 'md', color = 'purple' }) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16',
  }

  const colorClasses = {
    purple: 'border-purple-600',
    blue: 'border-blue-600',
    white: 'border-white',
  }

  return (
    <div
      className={`${sizeClasses[size]} ${colorClasses[color]} border-4 border-t-transparent rounded-full animate-spin`}
    />
  )
}

export default LoadingSpinner
