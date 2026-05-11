from ui import EmotionApp

if __name__ == "__main__":
    app = EmotionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()