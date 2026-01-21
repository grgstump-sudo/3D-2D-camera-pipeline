using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Diagnostics;
using System.Collections.Concurrent;

using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

using pho.api.csharp;

class Program
{
    // =========================
    // USER SETTINGS
    // =========================
    const int TARGET_FPS = 20;                 // target capture rate
    const double RUN_SECONDS = 10.0;           // change to 30 / 120 etc
    const int WRITER_THREADS = 2;              // disk writers
    const int QUEUE_CAPACITY = 120;            // buffering (frames)
    const int FRAME_WAIT_MS = 1200;            // wait for frame buffers to populate

    // IMPORTANT: default OFF so you don't get "blue bag"
    const bool ENABLE_WB = false;
    const bool ENABLE_BRIGHT_GAMMA = false;

    // For filtering ridiculous spikes (shouldn’t trigger once point cloud is correct)
    const float MAX_ABS_COORD = 1e7f;

    // =========================
    // EXPORT ORIENTATION FIX (CloudCompare upside-down)
    // =========================
    enum ExportOrientation
    {
        None,
        Rotate180_X,   // y=-y, z=-z  (most common "upside-down" fix)
        Rotate180_Y,   // x=-x, z=-z
        Rotate180_Z    // x=-x, y=-y
    }

    // Try Rotate180_X first. If it flips wrong, try Y or Z.
    const ExportOrientation EXPORT_ORIENTATION = ExportOrientation.Rotate180_X;

    // =========================
    // OUTPUT
    // =========================
    static readonly string OutDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
        "motioncam_color_png"
    );

    // =========================
    // MAIN
    // =========================
    static void Main()
    {
        Directory.CreateDirectory(OutDir);

        Console.WriteLine("OutDir:   " + OutDir);
        Console.WriteLine("Leave PhoXi Control running and device OPEN (Occupied is OK).");
        Console.WriteLine();

        var factory = new PhoXiFactory();
        PhoXi dev = factory.CreateAndConnectFirstAttached();

        if (dev == null || !dev.isConnected())
        {
            Console.WriteLine("Could not connect. Make sure the device is opened in PhoXi Control.");
            Console.ReadKey();
            return;
        }

        Console.WriteLine("Connected OK.");

        // Start acquisition if needed
        if (!dev.isAcquiring())
            dev.StartAcquisition();

        // Threading
        var queue = new BlockingCollection<FrameJob>(QUEUE_CAPACITY);
        WriterStats.Reset();

        var pngPool = new SimpleBytePool();
        var texPool = new SimpleBytePool();

        Thread[] writers = new Thread[WRITER_THREADS];
        for (int i = 0; i < WRITER_THREADS; i++)
        {
            writers[i] = new Thread(() => WriterLoop(queue, pngPool));
            writers[i].IsBackground = true;
            writers[i].Start();
        }

        // FPS pacing
        int targetFrameMs = (int)Math.Max(1, Math.Round(1000.0 / TARGET_FPS));
        Stopwatch sw = Stopwatch.StartNew();

        int captured = 0, enqueued = 0, dropped = 0, failed = 0, cloudOk = 0, cloudMiss = 0;

        // Reporting
        long lastReportMs = 0;
        int lastCaptured = 0, lastSavedPng = 0, lastSavedPly = 0;

        int seq = 0;

        // Optional WB (disabled by default)
        WbScales wb = new WbScales(1f, 1f, 1f);
        bool wbReady = false;

        // brightness/gamma LUT (disabled by default)
        byte[] lut = null;
        if (ENABLE_BRIGHT_GAMMA)
            lut = BuildSimpleGammaLut(1.15f, 1.0f);

        Console.WriteLine();
        Console.WriteLine($"Free-run: {RUN_SECONDS:0.##}s @ target {TARGET_FPS} fps (~{targetFrameMs} ms/frame)");
        Console.WriteLine($"Writers: {WRITER_THREADS}, QueueCapacity: {QUEUE_CAPACITY}");
        Console.WriteLine($"PLY export orientation: {EXPORT_ORIENTATION}");
        Console.WriteLine();

        double stopAt = RUN_SECONDS;

        while (sw.ElapsedMilliseconds / 1000.0 < stopAt)
        {
            long frameStartMs = sw.ElapsedMilliseconds;

            try
            {
                int idx = dev.TriggerFrame(true, true);
                if (idx < 0)
                {
                    failed++;
                    Pace(sw, frameStartMs, targetFrameMs);
                    continue;
                }

                // Wait for frame to become populated
                Frame frame = null;
                object colorMat = null;   // ColorCameraImage for PNG
                object texMat = null;     // TextureRGB / Texture for vertex colors
                object cloudMat = null;   // PointCloud

                Stopwatch wait = Stopwatch.StartNew();
                while (wait.ElapsedMilliseconds < FRAME_WAIT_MS)
                {
                    frame = dev.GetFrame(idx);
                    if (frame != null)
                    {
                        // 1) PNG: force ColorCameraImage first (keeps background & correct colors)
                        colorMat = GetFirstNonEmptyMat(frame, new string[]
                        {
                            "ColorCameraImage",
                            "ColorCameraImageRGB",
                            "ColorCameraImageRGB16",
                            "ColorCamera",
                            "TextureRGB",      // fallback only if color cam not available
                            "Texture"
                        }, out string chosenColor);

                        // 2) Texture (for coloring the point cloud) - prefer TextureRGB/Texture
                        texMat = GetFirstNonEmptyMat(frame, new string[]
                        {
                            "TextureRGB",
                            "Texture",
                            "ColorCameraImage" // last fallback
                        }, out string chosenTex);

                        // 3) Cloud: force PointCloud and use GetDataCopyXYZXYZ
                        cloudMat = GetFirstNonEmptyCloudMat(frame, out string chosenCloud);

                        if (colorMat != null && cloudMat != null)
                            break;
                    }

                    Thread.Sleep(5);
                }

                if (colorMat == null)
                {
                    failed++;
                    Pace(sw, frameStartMs, targetFrameMs);
                    continue;
                }

                // ---- Resolve COLOR mat dims / type
                bool colorIsRgb16 = IsRgb16Like(colorMat);
                if (!TryGetMatDimensions(colorMat, out int pngW, out int pngH))
                {
                    int bpp = colorIsRgb16 ? 6 : 3;
                    int len = GetDataCopyByteLengthPrefer(colorMat);
                    if (!TryInferDimensionsFromBytes(len, bpp, out pngW, out pngH))
                    {
                        failed++;
                        Pace(sw, frameStartMs, targetFrameMs);
                        continue;
                    }
                }

                // For MotionCam ColorCameraImage, BGR ordering is common
                bool pngSwapRB = (GetLastChosenNameWas(colorMat, "ColorCameraImage"));

                // ---- Resolve TEXTURE dims / type (optional)
                byte[] texRgb = null;
                int texW = 0, texH = 0;
                bool texIsRgb16 = false;
                bool texSwapRB = false;

                if (texMat != null)
                {
                    texIsRgb16 = IsRgb16Like(texMat);
                    if (!TryGetMatDimensions(texMat, out texW, out texH))
                    {
                        int bpp = texIsRgb16 ? 6 : 3;
                        int len = GetDataCopyByteLengthPrefer(texMat);
                        TryInferDimensionsFromBytes(len, bpp, out texW, out texH);
                    }
                    texSwapRB = false; // TextureRGB usually already in RGB
                }

                // allocate buffers
                byte[] pngRgb = pngPool.Rent(pngW * pngH * 3);

                if (texMat != null && texW > 0 && texH > 0)
                    texRgb = texPool.Rent(texW * texH * 3);

                // Convert mats
                ConvertMatToRgb8(colorMat, pngRgb, pngW, pngH, colorIsRgb16, pngSwapRB, rgb16Stretch: true);

                if (ENABLE_WB)
                {
                    if (!wbReady)
                    {
                        wb = ComputeWbScalesSampled(pngRgb, pngW, pngH, 0.85f, 8);
                        wbReady = true;
                        Console.WriteLine($"WB scales: R {wb.R:F3}, G {wb.G:F3}, B {wb.B:F3}");
                    }
                    ApplyWbScales(pngRgb, wb);
                }

                if (ENABLE_BRIGHT_GAMMA && lut != null)
                    ApplyLut(pngRgb, lut);

                if (texRgb != null)
                    ConvertMatToRgb8(texMat, texRgb, texW, texH, texIsRgb16, texSwapRB, rgb16Stretch: true);

                // ---- Get real XYZ point cloud (THE FIX)
                bool hasCloud = false;
                float[] cloudXYZ_valid = null;
                byte[] cloudRGB_valid = null;

                if (cloudMat != null)
                {
                    if (TryGetPointCloudXYZ_FromPointCloudObject(cloudMat, out float[] cloudXYZ_all, out int cloudW, out int cloudH))
                    {
                        // If we have texture that matches the cloud resolution, use it for vertex colors
                        byte[] rgbAll = null;
                        if (texRgb != null && cloudW > 0 && cloudH > 0 && texW == cloudW && texH == cloudH)
                            rgbAll = texRgb;

                        FilterValidPointsXYZ(cloudXYZ_all, rgbAll, out cloudXYZ_valid, out cloudRGB_valid);

                        if (cloudXYZ_valid != null && cloudXYZ_valid.Length >= 3)
                        {
                            hasCloud = true;
                            cloudOk++;
                        }
                        else cloudMiss++;
                    }
                    else cloudMiss++;
                }
                else cloudMiss++;

                // enqueue job
                seq++;
                captured++;

                string pngPath = Path.Combine(OutDir, "color_" + seq.ToString("000000") + ".png");
                string plyPath = Path.Combine(OutDir, "cloud_" + seq.ToString("000000") + ".ply");

                var job = new FrameJob
                {
                    Seq = seq,
                    PngW = pngW,
                    PngH = pngH,
                    PngRgb8 = pngRgb,
                    PngPath = pngPath,

                    HasCloud = hasCloud,
                    CloudXYZ = cloudXYZ_valid,
                    CloudRgb8 = cloudRGB_valid,
                    PlyPath = plyPath
                };

                if (!queue.TryAdd(job))
                {
                    dropped++;
                    pngPool.Return(pngRgb);
                    if (texRgb != null) texPool.Return(texRgb);
                }
                else
                {
                    enqueued++;
                    if (texRgb != null) texPool.Return(texRgb);
                }

                // report once/sec
                long now = sw.ElapsedMilliseconds;
                if (now - lastReportMs >= 1000)
                {
                    int savedPng = WriterStats.SavedPng;
                    int savedPly = WriterStats.SavedPly;

                    int capDelta = captured - lastCaptured;
                    int pngDelta = savedPng - lastSavedPng;
                    int plyDelta = savedPly - lastSavedPly;

                    Console.WriteLine(
                        "t=" + (now / 1000.0).ToString("F1") + "s"
                        + " | cap=" + captured + " (" + capDelta + "/s)"
                        + " | png=" + savedPng + " (" + pngDelta + "/s)"
                        + " | ply=" + savedPly + " (" + plyDelta + "/s)"
                        + " | q=" + queue.Count
                        + " | drop=" + dropped
                        + " | fail=" + failed
                        + " | cloud_ok=" + cloudOk
                        + " | cloud_miss=" + cloudMiss
                    );

                    lastReportMs = now;
                    lastCaptured = captured;
                    lastSavedPng = savedPng;
                    lastSavedPly = savedPly;
                }

                Pace(sw, frameStartMs, targetFrameMs);
            }
            catch
            {
                failed++;
                Pace(sw, frameStartMs, targetFrameMs);
            }
        }

        Console.WriteLine("\nStopping capture... draining queue...");
        queue.CompleteAdding();
        foreach (var t in writers) t.Join();

        dev.StopAcquisition();
        dev.Disconnect();

        double seconds = Math.Max(0.001, sw.ElapsedMilliseconds / 1000.0);
        Console.WriteLine("\nDone.");
        Console.WriteLine("Captured:  " + captured + " => " + (captured / seconds).ToString("F2") + " fps");
        Console.WriteLine("Saved PNG: " + WriterStats.SavedPng);
        Console.WriteLine("Saved PLY: " + WriterStats.SavedPly);
        Console.WriteLine("Dropped:   " + dropped);
        Console.WriteLine("Failed:    " + failed);
        Console.WriteLine("Cloud OK:  " + cloudOk);
        Console.WriteLine("Cloud Miss:" + cloudMiss);
        Console.WriteLine("Press any key to exit.");
        Console.ReadKey();
    }

    // ======================================================
    // Cloud extraction for PointCloud object
    // Uses GetDataCopyXYZXYZ() like Photoneo example code
    // ======================================================
    static bool TryGetPointCloudXYZ_FromPointCloudObject(object pointCloudObj, out float[] xyz, out int width, out int height)
    {
        xyz = null;
        width = 0;
        height = 0;
        if (pointCloudObj == null) return false;

        object copy = TryCallObjectMethod(pointCloudObj, "GetDataCopyXYZXYZ");
        if (copy == null)
        {
            copy = TryCallObjectMethod(pointCloudObj, "GetDataCopyXYZ");
        }

        if (copy is float[] f && f.Length >= 3)
            xyz = f;
        else
            return false;

        TryGetMatDimensions(pointCloudObj, out width, out height);

        if (xyz.Length % 3 != 0) return false;
        return true;
    }

    // ======================================================
    // Filter out invalid points
    // ======================================================
    static void FilterValidPointsXYZ(float[] xyzAll, byte[] rgbAllOrNull, out float[] xyzValid, out byte[] rgbValid)
    {
        xyzValid = null;
        rgbValid = null;

        if (xyzAll == null || xyzAll.Length < 3) return;

        int pointCount = xyzAll.Length / 3;
        bool hasRgb = (rgbAllOrNull != null && rgbAllOrNull.Length >= pointCount * 3);

        int valid = 0;
        for (int i = 0; i < pointCount; i++)
        {
            int bi = i * 3;
            float x = xyzAll[bi + 0];
            float y = xyzAll[bi + 1];
            float z = xyzAll[bi + 2];
            if (!IsValidPoint(x, y, z)) continue;
            valid++;
        }
        if (valid <= 0) return;

        xyzValid = new float[valid * 3];
        if (hasRgb) rgbValid = new byte[valid * 3];

        int v = 0;
        for (int i = 0; i < pointCount; i++)
        {
            int bi = i * 3;
            float x = xyzAll[bi + 0];
            float y = xyzAll[bi + 1];
            float z = xyzAll[bi + 2];

            if (!IsValidPoint(x, y, z)) continue;

            int vi = v * 3;
            xyzValid[vi + 0] = x;
            xyzValid[vi + 1] = y;
            xyzValid[vi + 2] = z;

            if (hasRgb)
            {
                int ri = i * 3;
                rgbValid[vi + 0] = rgbAllOrNull[ri + 0];
                rgbValid[vi + 1] = rgbAllOrNull[ri + 1];
                rgbValid[vi + 2] = rgbAllOrNull[ri + 2];
            }

            v++;
        }
    }

    static bool IsValidPoint(float x, float y, float z)
    {
        if (float.IsNaN(x) || float.IsNaN(y) || float.IsNaN(z)) return false;
        if (float.IsInfinity(x) || float.IsInfinity(y) || float.IsInfinity(z)) return false;
        if (x == 0f && y == 0f && z == 0f) return false;
        if (Math.Abs(x) > MAX_ABS_COORD || Math.Abs(y) > MAX_ABS_COORD || Math.Abs(z) > MAX_ABS_COORD) return false;
        return true;
    }

    // ======================================================
    // MAT / FRAME helpers
    // ======================================================
    static object GetFirstNonEmptyMat(Frame frame, string[] propsToTry, out string chosenName)
    {
        chosenName = null;
        if (frame == null) return null;

        foreach (var name in propsToTry)
        {
            object obj = TryGetProp(frame, name);
            if (obj == null) continue;

            int len = GetDataCopyByteLengthPrefer(obj);
            if (len > 0)
            {
                chosenName = name;
                LastChosenName = name;
                return obj;
            }
        }
        return null;
    }

    static object GetFirstNonEmptyCloudMat(Frame frame, out string chosenName)
    {
        chosenName = null;
        if (frame == null) return null;

        string[] propsToTry = new string[] { "PointCloud", "PointCloud32f" };

        foreach (var name in propsToTry)
        {
            object obj = TryGetProp(frame, name);
            if (obj == null) continue;

            int len = GetDataCopyByteLengthPrefer(obj);
            if (len > 0)
            {
                chosenName = name;
                return obj;
            }
        }
        return null;
    }

    static object TryGetProp(object obj, string propName)
    {
        if (obj == null) return null;
        var t = obj.GetType();
        var p = t.GetProperty(propName, BindingFlags.Public | BindingFlags.Instance);
        if (p == null) return null;
        try { return p.GetValue(obj, null); } catch { return null; }
    }

    static object TryCallObjectMethod(object obj, string methodName)
    {
        if (obj == null) return null;
        var t = obj.GetType();
        var m = t.GetMethod(methodName, BindingFlags.Public | BindingFlags.Instance);
        if (m == null) return null;
        try { return m.Invoke(obj, null); } catch { return null; }
    }

    static int GetDataCopyByteLengthPrefer(object matObj)
    {
        if (matObj == null) return 0;

        var t = matObj.GetType();
        var mxyz = t.GetMethod("GetDataCopyXYZXYZ", BindingFlags.Public | BindingFlags.Instance);
        if (mxyz != null)
        {
            try
            {
                object v = mxyz.Invoke(matObj, null);
                if (v is Array a) return Buffer.ByteLength(a);
            }
            catch { }
        }

        var m = t.GetMethod("GetDataCopy", BindingFlags.Public | BindingFlags.Instance);
        if (m == null) return 0;

        try
        {
            object v = m.Invoke(matObj, null);
            if (v is Array a) return Buffer.ByteLength(a);
        }
        catch { }
        return 0;
    }

    static bool TryGetMatDimensions(object matObj, out int width, out int height)
    {
        width = 0; height = 0;
        if (matObj == null) return false;

        try
        {
            var sizeProp = matObj.GetType().GetProperty("Size", BindingFlags.Public | BindingFlags.Instance);
            if (sizeProp != null)
            {
                object sizeObj = sizeProp.GetValue(matObj, null);
                if (sizeObj != null)
                {
                    var wProp = sizeObj.GetType().GetProperty("Width");
                    var hProp = sizeObj.GetType().GetProperty("Height");
                    if (wProp != null && hProp != null)
                    {
                        width = (int)wProp.GetValue(sizeObj, null);
                        height = (int)hProp.GetValue(sizeObj, null);
                        return (width > 0 && height > 0);
                    }
                }
            }

            var wp = matObj.GetType().GetProperty("Width");
            var hp = matObj.GetType().GetProperty("Height");
            if (wp != null && hp != null)
            {
                width = (int)wp.GetValue(matObj, null);
                height = (int)hp.GetValue(matObj, null);
                return (width > 0 && height > 0);
            }
        }
        catch { }

        return false;
    }

    static bool TryInferDimensionsFromBytes(int byteLen, int bytesPerPixel, out int width, out int height)
    {
        width = 0; height = 0;
        if (byteLen <= 0 || bytesPerPixel <= 0) return false;

        int[] widths = new[] { 1932, 1920, 1680, 1600, 1440, 1280, 1024, 800, 640 };
        foreach (int w in widths)
        {
            int pixels = byteLen / bytesPerPixel;
            if (pixels % w != 0) continue;
            int h = pixels / w;
            if (h > 0)
            {
                width = w;
                height = h;
                return true;
            }
        }
        return false;
    }

    static bool IsRgb16Like(object mat)
    {
        if (mat == null) return false;
        string tn = (mat.GetType().FullName ?? mat.GetType().Name ?? "").ToLowerInvariant();
        return tn.Contains("rgb16") || tn.Contains("texturergb16");
    }

    static string LastChosenName = null;
    static bool GetLastChosenNameWas(object _unused, string name) => (LastChosenName == name);

    // ======================================================
    // COLOR conversion helpers
    // ======================================================
    static void ConvertMatToRgb8(object matObj, byte[] rgbOut, int width, int height, bool isRgb16, bool swapRB, bool rgb16Stretch)
    {
        object copyObj = TryCallObjectMethod(matObj, "GetDataCopy");
        if (copyObj == null) throw new Exception("GetDataCopy() returned null.");
        Array arr = copyObj as Array;
        if (arr == null || arr.Length == 0) throw new Exception("GetDataCopy() did not return an array.");

        if (isRgb16 && arr is ushort[] us)
        {
            ConvertUShortRgbToRgb8(us, rgbOut, width, height, swapRB, rgb16Stretch);
            return;
        }

        byte[] raw = ToBytesFromPrimitiveArray(arr);

        if (isRgb16)
        {
            int pixels = width * height;
            int neededBytes = pixels * 6;
            int usable = Math.Min(raw.Length, neededBytes);

            ushort[] u = new ushort[usable / 2];
            Buffer.BlockCopy(raw, 0, u, 0, u.Length * 2);
            ConvertUShortRgbToRgb8(u, rgbOut, width, height, swapRB, rgb16Stretch);
        }
        else
        {
            int pixels = width * height;
            int neededBytes = pixels * 3;
            int usable = Math.Min(raw.Length, neededBytes);

            if (!swapRB)
            {
                Buffer.BlockCopy(raw, 0, rgbOut, 0, usable);
            }
            else
            {
                for (int i = 0; i < usable; i += 3)
                {
                    byte b = raw[i + 0];
                    byte g = raw[i + 1];
                    byte r = raw[i + 2];
                    rgbOut[i + 0] = r;
                    rgbOut[i + 1] = g;
                    rgbOut[i + 2] = b;
                }
            }
        }
    }

    static byte[] ToBytesFromPrimitiveArray(Array arr)
    {
        if (arr is byte[] b) return b;
        if (arr is short[] s) { byte[] o = new byte[s.Length * 2]; Buffer.BlockCopy(s, 0, o, 0, o.Length); return o; }
        if (arr is ushort[] us) { byte[] o = new byte[us.Length * 2]; Buffer.BlockCopy(us, 0, o, 0, o.Length); return o; }
        if (arr is int[] i) { byte[] o = new byte[i.Length * 4]; Buffer.BlockCopy(i, 0, o, 0, o.Length); return o; }
        if (arr is uint[] ui) { byte[] o = new byte[ui.Length * 4]; Buffer.BlockCopy(ui, 0, o, 0, o.Length); return o; }
        if (arr is float[] f) { byte[] o = new byte[f.Length * 4]; Buffer.BlockCopy(f, 0, o, 0, o.Length); return o; }
        if (arr is double[] d) { byte[] o = new byte[d.Length * 8]; Buffer.BlockCopy(d, 0, o, 0, o.Length); return o; }

        try
        {
            int bl = Buffer.ByteLength(arr);
            byte[] raw = new byte[bl];
            Buffer.BlockCopy(arr, 0, raw, 0, bl);
            return raw;
        }
        catch
        {
            return null;
        }
    }

    static void ConvertUShortRgbToRgb8(ushort[] src, byte[] dst, int width, int height, bool swapRB, bool stretch)
    {
        int pixels = Math.Min(width * height, src.Length / 3);
        if (pixels <= 0) return;

        if (!stretch)
        {
            for (int i = 0; i < pixels; i++)
            {
                int si = i * 3;
                int di = i * 3;
                ushort r16 = src[si + 0];
                ushort g16 = src[si + 1];
                ushort b16 = src[si + 2];

                byte r = (byte)(r16 >> 8);
                byte g = (byte)(g16 >> 8);
                byte b = (byte)(b16 >> 8);

                if (!swapRB)
                {
                    dst[di + 0] = r;
                    dst[di + 1] = g;
                    dst[di + 2] = b;
                }
                else
                {
                    dst[di + 0] = b;
                    dst[di + 1] = g;
                    dst[di + 2] = r;
                }
            }
        }
        else
        {
            ushort rmin = ushort.MaxValue, gmin = ushort.MaxValue, bmin = ushort.MaxValue;
            ushort rmax = 0, gmax = 0, bmax = 0;

            for (int i = 0; i < pixels; i++)
            {
                int si = i * 3;
                ushort r = src[si + 0], g = src[si + 1], b = src[si + 2];
                if (r < rmin) rmin = r; if (r > rmax) rmax = r;
                if (g < gmin) gmin = g; if (g > gmax) gmax = g;
                if (b < bmin) bmin = b; if (b > bmax) bmax = b;
            }

            float rs = (rmax > rmin) ? (255f / (rmax - rmin)) : 1f;
            float gs = (gmax > gmin) ? (255f / (gmax - gmin)) : 1f;
            float bs = (bmax > bmin) ? (255f / (bmax - bmin)) : 1f;

            for (int i = 0; i < pixels; i++)
            {
                int si = i * 3;
                int di = i * 3;

                byte r8 = (byte)Math.Max(0, Math.Min(255, (src[si + 0] - rmin) * rs));
                byte g8 = (byte)Math.Max(0, Math.Min(255, (src[si + 1] - gmin) * gs));
                byte b8 = (byte)Math.Max(0, Math.Min(255, (src[si + 2] - bmin) * bs));

                if (!swapRB)
                {
                    dst[di + 0] = r8;
                    dst[di + 1] = g8;
                    dst[di + 2] = b8;
                }
                else
                {
                    dst[di + 0] = b8;
                    dst[di + 1] = g8;
                    dst[di + 2] = r8;
                }
            }
        }
    }

    // ======================================================
    // Writers
    // ======================================================
    struct FrameJob
    {
        public int Seq;

        public int PngW, PngH;
        public byte[] PngRgb8;
        public string PngPath;

        public bool HasCloud;
        public float[] CloudXYZ;
        public byte[] CloudRgb8; // can be null
        public string PlyPath;
    }

    static class WriterStats
    {
        static int _png = 0;
        static int _ply = 0;
        public static int SavedPng => _png;
        public static int SavedPly => _ply;
        public static void Reset() { _png = 0; _ply = 0; }
        public static void IncPng() { Interlocked.Increment(ref _png); }
        public static void IncPly() { Interlocked.Increment(ref _ply); }
    }

    static void WriterLoop(BlockingCollection<FrameJob> queue, SimpleBytePool pngPool)
    {
        foreach (var job in queue.GetConsumingEnumerable())
        {
            try
            {
                WritePng(job.PngPath, job.PngW, job.PngH, job.PngRgb8);
                WriterStats.IncPng();
            }
            catch { }

            try
            {
                if (job.HasCloud && job.CloudXYZ != null && job.CloudXYZ.Length >= 3)
                {
                    WriteBinaryPly(job.PlyPath, job.CloudXYZ, job.CloudRgb8);
                    WriterStats.IncPly();
                }
            }
            catch { }

            if (job.PngRgb8 != null)
                pngPool.Return(job.PngRgb8);
        }
    }

    // Real PNG output (NO PPM)
    // rgb[] is RGBRGB... (24bpp)
    // Bitmap wants BGR per pixel for Format24bppRgb memory layout.
    static void WritePng(string path, int w, int h, byte[] rgb)
    {
        // Ensure .png extension
        if (!path.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            path = Path.ChangeExtension(path, ".png");

        using (var bmp = new Bitmap(w, h, PixelFormat.Format24bppRgb))
        {
            var rect = new Rectangle(0, 0, w, h);
            BitmapData data = bmp.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

            try
            {
                int stride = data.Stride;
                IntPtr scan0 = data.Scan0;

                // Build a packed BGR buffer respecting stride
                byte[] bgr = new byte[stride * h];

                int srcRowBytes = w * 3;
                for (int y = 0; y < h; y++)
                {
                    int srcRow = y * srcRowBytes;
                    int dstRow = y * stride;

                    // RGB -> BGR
                    for (int x = 0; x < w; x++)
                    {
                        int si = srcRow + x * 3;
                        int di = dstRow + x * 3;

                        byte r = rgb[si + 0];
                        byte g = rgb[si + 1];
                        byte b = rgb[si + 2];

                        bgr[di + 0] = b;
                        bgr[di + 1] = g;
                        bgr[di + 2] = r;
                    }
                }

                Marshal.Copy(bgr, 0, scan0, bgr.Length);
            }
            finally
            {
                bmp.UnlockBits(data);
            }

            bmp.Save(path, ImageFormat.Png);
        }
    }

    static void WriteBinaryPly(string path, float[] xyz, byte[] rgbOrNull)
    {
        int n = xyz.Length / 3;
        bool hasRgb = (rgbOrNull != null && rgbOrNull.Length >= xyz.Length);

        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write(System.Text.Encoding.ASCII.GetBytes("ply\n"));
            bw.Write(System.Text.Encoding.ASCII.GetBytes("format binary_little_endian 1.0\n"));
            bw.Write(System.Text.Encoding.ASCII.GetBytes($"element vertex {n}\n"));
            bw.Write(System.Text.Encoding.ASCII.GetBytes("property float x\n"));
            bw.Write(System.Text.Encoding.ASCII.GetBytes("property float y\n"));
            bw.Write(System.Text.Encoding.ASCII.GetBytes("property float z\n"));
            bw.Write(System.Text.Encoding.ASCII.GetBytes("property uchar red\n"));
            bw.Write(System.Text.Encoding.ASCII.GetBytes("property uchar green\n"));
            bw.Write(System.Text.Encoding.ASCII.GetBytes("property uchar blue\n"));
            bw.Write(System.Text.Encoding.ASCII.GetBytes("end_header\n"));

            for (int i = 0; i < n; i++)
            {
                int bi = i * 3;

                float x = xyz[bi + 0];
                float y = xyz[bi + 1];
                float z = xyz[bi + 2];

                // Orientation fix at export-time
                switch (EXPORT_ORIENTATION)
                {
                    case ExportOrientation.Rotate180_X:
                        y = -y; z = -z;
                        break;
                    case ExportOrientation.Rotate180_Y:
                        x = -x; z = -z;
                        break;
                    case ExportOrientation.Rotate180_Z:
                        x = -x; y = -y;
                        break;
                    case ExportOrientation.None:
                    default:
                        break;
                }

                bw.Write(x);
                bw.Write(y);
                bw.Write(z);

                if (hasRgb)
                {
                    bw.Write(rgbOrNull[bi + 0]);
                    bw.Write(rgbOrNull[bi + 1]);
                    bw.Write(rgbOrNull[bi + 2]);
                }
                else
                {
                    bw.Write((byte)255);
                    bw.Write((byte)255);
                    bw.Write((byte)255);
                }
            }
        }
    }

    // ======================================================
    // Pools + misc
    // ======================================================
    class SimpleBytePool
    {
        ConcurrentBag<byte[]> bag = new ConcurrentBag<byte[]>();

        public byte[] Rent(int minSize = 0)
        {
            if (bag.TryTake(out var b))
            {
                if (b.Length >= minSize) return b;
            }
            return new byte[Math.Max(1, minSize)];
        }

        public void Return(byte[] b)
        {
            if (b == null) return;
            bag.Add(b);
        }
    }

    static void Pace(Stopwatch sw, long frameStartMs, int targetFrameMs)
    {
        long elapsed = sw.ElapsedMilliseconds - frameStartMs;
        int remaining = (int)(targetFrameMs - elapsed);
        if (remaining > 0) Thread.Sleep(remaining);
    }

    // ======================================================
    // WB / LUT (optional)
    // ======================================================
    struct WbScales { public float R, G, B; public WbScales(float r, float g, float b) { R = r; G = g; B = b; } }

    static WbScales ComputeWbScalesSampled(byte[] rgb, int w, int h, float strength, int stride)
    {
        long r = 0, g = 0, b = 0; long c = 0;
        int step = Math.Max(1, stride);

        for (int y = 0; y < h; y += step)
        {
            int row = y * w * 3;
            for (int x = 0; x < w; x += step)
            {
                int i = row + x * 3;
                r += rgb[i + 0];
                g += rgb[i + 1];
                b += rgb[i + 2];
                c++;
            }
        }
        if (c <= 0) return new WbScales(1f, 1f, 1f);

        float rf = (float)r / c, gf = (float)g / c, bf = (float)b / c;
        float gray = (rf + gf + bf) / 3f;

        float sr = (rf > 1e-3f) ? gray / rf : 1f;
        float sg = (gf > 1e-3f) ? gray / gf : 1f;
        float sb = (bf > 1e-3f) ? gray / bf : 1f;

        sr = 1f + (sr - 1f) * strength;
        sg = 1f + (sg - 1f) * strength;
        sb = 1f + (sb - 1f) * strength;

        return new WbScales(sr, sg, sb);
    }

    static void ApplyWbScales(byte[] rgb, WbScales s)
    {
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i + 0] = ClampByte(rgb[i + 0] * s.R);
            rgb[i + 1] = ClampByte(rgb[i + 1] * s.G);
            rgb[i + 2] = ClampByte(rgb[i + 2] * s.B);
        }
    }

    static byte ClampByte(float v)
    {
        if (v < 0f) return 0;
        if (v > 255f) return 255;
        return (byte)(v + 0.5f);
    }

    static byte[] BuildSimpleGammaLut(float gain, float gamma)
    {
        byte[] lut = new byte[256];
        for (int i = 0; i < 256; i++)
        {
            float x = i / 255f;
            x = (float)Math.Pow(x, gamma);
            x *= gain;
            if (x < 0f) x = 0f;
            if (x > 1f) x = 1f;
            lut[i] = (byte)(x * 255f + 0.5f);
        }
        return lut;
    }

    static void ApplyLut(byte[] rgb, byte[] lut)
    {
        if (lut == null || lut.Length != 256) return;
        for (int i = 0; i < rgb.Length; i++)
            rgb[i] = lut[rgb[i]];
    }
}
