import os
import ctypes
os.environ["PYOPENGL_PLATFORM"] = "egl"

try:
    from OpenGL.EGL import eglGetDisplay, EGL_DEFAULT_DISPLAY, eglInitialize
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    
    if display:
        # Create C-style integer pointers to hold the version numbers
        major = ctypes.c_int()
        minor = ctypes.c_int()
        
        
        # Pass all 3 arguments exactly as the C library expects
        success = eglInitialize(display, major, minor)
        
        if success:
            print(f"✅ SUCCESS! EGL is working. Version: {major.value}.{minor.value}")
        else:
            print("❌ FAILED to initialize the EGL display.")
    else:
        print("❌ FAILED. display is None.")
except Exception as e:
    print(f"❌ CRASH: {e}")