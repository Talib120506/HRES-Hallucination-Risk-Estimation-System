# HRES Frontend - React + Vite

Modern React frontend for the Hallucination Risk Estimation System (HRES).

## Features

- **Beautiful Landing Page**: Hero section, features, how it works, and CTA
- **Interactive App Interface**:
  - Tab-based UI for preloaded PDFs or file upload
  - Drag-and-drop PDF upload
  - Real-time form validation
- **Results Visualization**:
  - Combined verdict with agreement logic
  - Expandable sections for whitebox and blackbox details
  - Color-coded verdict cards with progress bars
- **Responsive Design**: Mobile-first, works on all screen sizes
- **Modern UI**: Tailwind CSS with purple-blue gradient theme

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Environment Configuration

The `.env` file is already configured:

```
VITE_API_URL=http://localhost:8000
```

## Running the Development Server

```bash
cd frontend
npm run dev
```

The app will be available at `http://localhost:5173`

## Building for Production

```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── landing/          # Landing page components
│   │   │   ├── HeroSection.jsx
│   │   │   ├── AboutSection.jsx
│   │   │   ├── FeaturesSection.jsx
│   │   │   ├── HowItWorksSection.jsx
│   │   │   └── CTASection.jsx
│   │   ├── app/              # Main app components
│   │   │   ├── Header.jsx
│   │   │   ├── TabContainer.jsx
│   │   │   ├── PreloadedTab.jsx
│   │   │   ├── UploadTab.jsx
│   │   │   ├── VerdictCard.jsx
│   │   │   └── ResultsPanel.jsx
│   │   └── common/           # Shared components
│   │       └── LoadingSpinner.jsx
│   ├── pages/
│   │   ├── LandingPage.jsx   # Home page (/)
│   │   └── AppPage.jsx       # Main app (/app)
│   ├── services/
│   │   └── api.js            # API client
│   ├── utils/
│   │   └── colors.js         # Verdict color utilities
│   ├── styles/
│   ├── App.jsx               # Main app with routing
│   ├── main.jsx              # Entry point
│   └── index.css             # Global styles + Tailwind
├── public/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## Routes

- `/` - Landing page with features and how it works
- `/app` - Main application for hallucination detection

## API Integration

The frontend communicates with the FastAPI backend through:

- `GET /api/preloaded-pdfs` - Fetch available PDFs
- `POST /api/predict` - Submit analysis request

See `src/services/api.js` for implementation details.

## Styling

Built with Tailwind CSS:

- **Primary Gradient**: Purple (#7c3aed) to Blue (#3b82f6)
- **Verdict Colors**:
  - Green: Correct/Grounded
  - Red: Hallucinated/Hallucination
  - Yellow: Uncertain
- **Responsive Breakpoints**: sm, md, lg, xl

## Key Features

### Landing Page

- Hero section with gradient background
- About section explaining the problem and solution
- 6 feature cards with icons
- Dual-pipeline explanation with visual flow
- Call-to-action section

### App Page

- Two-tab interface (Preloaded PDFs / Upload PDF)
- Form validation with error messages
- Loading overlay during analysis
- Success toast notification
- Expandable results with accordion sections
- Combined verdict with agreement logic
- Example cards for quick testing

### Verdict Card

- Color-coded borders and backgrounds
- Confidence scores with progress bars
- Probability breakdowns
- Retrieved context display (blackbox only)

## Development

### Hot Module Replacement

Vite provides instant HMR for rapid development.

### Proxy Configuration

API requests are proxied through Vite dev server to avoid CORS issues.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Troubleshooting

### Backend connection fails

- Ensure backend is running on port 8000
- Check VITE_API_URL in `.env` file

### Styles not loading

- Run `npm install` to ensure Tailwind CSS is installed
- Verify `tailwind.config.js` and `postcss.config.js` exist

### Build fails

- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Clear Vite cache: `rm -rf node_modules/.vite`
