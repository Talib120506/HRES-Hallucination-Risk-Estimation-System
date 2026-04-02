/**
 * API Service
 * Handles all API calls to the FastAPI backend
 */
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Fetch list of preloaded PDFs
 */
export const fetchPreloadedPDFs = async () => {
  try {
    const response = await api.get('/api/preloaded-pdfs')
    return response.data
  } catch (error) {
    console.warn('Error fetching preloaded PDFs, returning fallbacks:', error)
    // Fallback data when backend is not running
    return [
      { filename: 'bosch_oven.pdf', display_name: 'Bosch Oven Manual' },
      { filename: 'dewalt_saw.pdf', display_name: 'DeWalt Saw Manual' },
      { filename: 'dyson_v12.pdf', display_name: 'Dyson V12 Manual' },
      { filename: 'electrolux_oven.pdf', display_name: 'Electrolux Oven Manual' },
      { filename: 'hilti_hammer.pdf', display_name: 'Hilti Hammer Manual' },
      { filename: 'laptop_lenovo_tc_x1.pdf', display_name: 'Lenovo ThinkPad X1 Manual' },
      { filename: 'makita_drill.pdf', display_name: 'Makita Drill Manual' },
      { filename: 'omron_monitor.pdf', display_name: 'Omron Monitor Manual' },
      { filename: 'prusa_3d-printer.pdf', display_name: 'Prusa 3D Printer Manual' },
      { filename: 'spot_boston_dynamics.pdf', display_name: 'Spot Boston Dynamics Manual' }
    ]
  }
}

/**
 * Predict hallucination using uploaded file or preloaded PDF
 * @param {FormData} formData - Must contain: question, answer, and either file or preloaded_pdf
 */
export const predictHallucination = async (formData) => {
  try {
    const response = await api.post('/api/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    console.error('Error predicting hallucination:', error)
    if (error.code === 'ERR_NETWORK') {
      throw new Error('Cannot connect to server. Please make sure the backend is running on port 8000.')
    }
    throw new Error(error.response?.data?.detail || 'Failed to analyze answer')
  }
}

export default api
