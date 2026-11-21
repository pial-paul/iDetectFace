# Deploy the browser demo for free

## Try a check-in

Open the [live demo](https://idetectface.streamlit.app/) and choose English or বাংলা. Read how your photos will be used, then agree to begin.

1. **Add your first photo.** Take a clear picture of yourself or choose the upload option. The app confirms it found one face and advances automatically.
2. **Add a second photo.** Take a fresh picture or upload a different photo of yourself. The app compares it with your first photo.
3. **Read your result.** A match records your first check-in time for that Bangladesh calendar day (UTC+6). Download the CSV or try again to see daily duplicate prevention. Expand the match explanation to see the measured distance. Clear the session when finished.

## Hosting

The demo uses the desktop project's dlib detector, five-point landmark model, 128-dimensional encoding and strict 0.5 distance threshold. Visitors take snapshots or upload two photos of themselves. The server processes photos in memory and gives each browser session its own reference embedding and Bangladesh-time attendance register. Download the CSV before clearing the session. The desktop application supports continuous webcam recognition and persistent local records.

1. Open the [prefilled deployment form](https://share.streamlit.io/deploy?repository=pial-paul%2FiDetectFace&branch=main&mainModule=demo%2Fstreamlit_app.py), sign in to Streamlit Community Cloud and connect your GitHub account.
2. Select **Create app**, then **Yup, I have an app**. Select repository `pial-paul/iDetectFace`, branch `main`, and entrypoint `demo/streamlit_app.py`.
3. In **Advanced settings**, select **Python 3.11**. The app uses the requirements beside its entrypoint and needs no API keys. Choose an available subdomain and select **Deploy**.
4. Once the build completes, open the assigned `streamlit.app` URL in a private browser window. Allow photo processing, enroll with one photo, then check in twice with a second photo. Verify the downloaded CSV has one row and clear the session. If the browser blocks camera access, choose **Upload**.

Streamlit Community Cloud offers free hosting and can put an idle app to sleep. [Hosting overview](https://docs.streamlit.io/deploy/streamlit-community-cloud) · [Deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

To run from the repository root in a fresh Python 3.11 environment:

```sh
python -m pip install -r demo/requirements.txt
python -m unittest discover -s demo/tests -v
python -m streamlit run demo/streamlit_app.py
```
