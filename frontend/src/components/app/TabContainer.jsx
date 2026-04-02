function TabContainer({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'preloaded', label: 'Preloaded Documents' },
    { id: 'upload', label: 'Upload PDF' },
  ]

  return (
    <div className="flex border-b border-gray-100 dark:border-gray-800/50 relative">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          className={`relative flex-1 py-4 font-semibold transition-all duration-500 text-center z-10 group overflow-hidden ${
            activeTab === tab.id
              ? 'text-purple-600 dark:text-purple-400 bg-purple-50/50 dark:bg-purple-900/10 scale-[1.02] shadow-sm transform -translate-y-0.5'
              : 'text-gray-500 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-50/50 dark:hover:bg-gray-800/30 hover:scale-[1.01]'
          }`}
        >
          <span className="relative z-10">{tab.label}</span>
          
          {/* Hover Sweep Effect */}
          {activeTab !== tab.id && (
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/5 to-blue-500/0 transform -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
          )}

          {activeTab === tab.id && (
            <div className="absolute bottom-0 left-0 w-full h-[3px] bg-gradient-to-r from-purple-500 to-blue-500 shadow-[0_-2px_10px_rgba(168,85,247,0.5)] transform origin-left animate-[scaleX_0.3s_ease-out_forwards]"></div>
          )}
        </button>
      ))}
    </div>
  )
}

export default TabContainer
