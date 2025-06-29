from backend.app import app

if __name__ == "__main__":
    print("Starting Flask server via run.py...")
    # use_reloader should be False to prevent issues with background threads
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)