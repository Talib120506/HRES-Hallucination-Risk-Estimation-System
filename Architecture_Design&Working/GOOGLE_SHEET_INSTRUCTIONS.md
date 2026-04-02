### How to connect the "Get in Touch" form to Google Sheets

I have updated your frontend form to send data automatically if you provide a Google Apps Script url. Check out your `src/components/landing/ContactSection.jsx`!

Here is how you actually create that script and connect it:

1. **Create a Google Sheet**:
   - Go to Google Sheets and make a blank spreadsheet.
   - Name it "HRES Leads" (or whatever you'd like).
   - In row 1, set up these exactly four columns: `Name` | `Email` | `Message` | `Date`

2. **Open Apps Script**:
   - In your Google Sheet, click `Extensions` -> `Apps Script` in the top toolbar.

3. **Paste the Webhook Code**:
   - Delete the empty `myFunction` block and paste the following code exactly as is:

```javascript
function doPost(e) {
  try {
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    var rowData = [];

    // We are pulling data corresponding to the 'name' attributes on our form inputs
    rowData.push(e.parameter.name || "Unknown");
    rowData.push(e.parameter.email || "Unknown");
    rowData.push(e.parameter.message || "Unknown");
    rowData.push(new Date()); // Automatically add the current timestamp

    sheet.appendRow(rowData);

    return ContentService.createTextOutput(
      JSON.stringify({ result: "success" }),
    ).setMimeType(ContentService.MimeType.JSON);
  } catch (error) {
    return ContentService.createTextOutput(
      JSON.stringify({ result: "error", error: error.message }),
    ).setMimeType(ContentService.MimeType.JSON);
  }
}
```

4. **Deploy as a Web App**:
   - Click the blue **"Deploy"** button in the top right.
   - Choose **"New deployment"**.
   - Under "Select type" click the gear icon and select **"Web app"**.
   - Set "Execute as" to: **Me** (your email).
   - Set "Who has access" to: **Anyone** (This allows public users on your site to trigger it without logging into google).
   - Click **Deploy**. (It will ask you to authorize permissions—allow them to edit your sheets).

5. **Copy the Web App URL**:
   - Google will give you a long URL starting with `https://script.google.com/macros/s/....`
   - Copy this URL.

6. **Add the URL to your Code**:
   - Go back into your project, open `frontend/src/components/landing/ContactSection.jsx`
   - Look for line `10`: `const GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec";`
   - Replace that dummy URL string with your brand new copied url string.

That's it! Your landing page React form will seamlessly ping Google's servers using native `FormData` fetching, show your user an animated loading state while processing, and show them a success text once Google replies. The answers will line up beautifully on your sheet!
