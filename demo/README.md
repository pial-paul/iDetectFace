# Deploy the browser demo for free

The demo uses the desktop project's dlib detector, five-point landmark model, 128-dimensional encoding and strict 0.5 distance threshold. Visitors take snapshots or upload two photos of themselves. The server processes photos in memory and gives each browser session its own reference embedding and UTC attendance register. Download the CSV before clearing the session. The desktop application supports continuous webcam recognition and persistent local records.

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
