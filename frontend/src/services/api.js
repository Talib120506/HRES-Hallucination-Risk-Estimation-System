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
      { filename: 'apple_watch.pdf', display_name: 'Apple Watch' },
      { filename: 'bosch_oven.pdf', display_name: 'Bosch Oven' },
      { filename: 'dewalt_saw.pdf', display_name: 'DeWalt Saw' },
      { filename: 'dji_mavic_pro.pdf', display_name: 'DJI Mavic Pro' },
      { filename: 'dyson_v12.pdf', display_name: 'Dyson V12' },
      { filename: 'electrolux_oven.pdf', display_name: 'Electrolux Oven' },
      { filename: 'ford_mach_e.pdf', display_name: 'Ford Mach-E' },
      { filename: 'hilti_hammer.pdf', display_name: 'Hilti Hammer' },
      { filename: 'laptop_lenovo_tc_x1.pdf', display_name: 'Lenovo ThinkPad X1' },
      { filename: 'lg_home_theater.pdf', display_name: 'LG Home Theater' },
      { filename: 'makita_drill.pdf', display_name: 'Makita Drill' },
      { filename: 'nintendo_2ds_xl.pdf', display_name: 'Nintendo 2DS XL' },
      { filename: 'omron_monitor.pdf', display_name: 'Omron Monitor' },
      { filename: 'prusa_3d-printer.pdf', display_name: 'Prusa 3D Printer' },
      { filename: 'samsung_phone_zfold.pdf', display_name: 'Samsung Z Fold' },
      { filename: 'spot_boston_dynamics.pdf', display_name: 'Spot Boston Dynamics' },
      { filename: 'tesla_model_s.pdf', display_name: 'Tesla Model S' }
    ]
  }
}

/**
 * Predict hallucination using uploaded file or preloaded PDF (manual answer input)
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

/**
 * Generate answer using AI and then verify it (AI Generate Answer mode)
 * @param {FormData} formData - Must contain: question, and either file or preloaded_pdf
 */
export const generateAndVerify = async (formData) => {
  try {
    const response = await api.post('/api/generate-and-verify', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    console.error('Error generating and verifying answer:', error)
    if (error.code === 'ERR_NETWORK') {
      throw new Error('Cannot connect to server. Please make sure the backend is running on port 8000.')
    }
    throw new Error(error.response?.data?.detail || 'Failed to generate and analyze answer')
  }
}

export default api
