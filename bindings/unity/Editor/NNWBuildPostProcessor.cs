using UnityEngine;
using UnityEditor;
using UnityEditor.Callbacks;
using System.IO;

public class NNWBuildPostProcessor
{
    [PostProcessBuild(1)]
    public static void OnPostProcessBuild(BuildTarget target, string pathToBuiltProject)
    {
        if (target != BuildTarget.WebGL) return;

        var sourceDir = FindPluginsWebGLDir();
        if (sourceDir == null)
        {
            Debug.LogError("[NNW] Could not locate Runtime/Plugins/WebGL inside the package.");
            return;
        }

        var destDir = Path.Combine(pathToBuiltProject, "StreamingAssets", "NNW");
        Directory.CreateDirectory(destDir);

        CopyIfExists(sourceDir, destDir, "nn.js");
        CopyIfExists(sourceDir, destDir, "nn.wasm");

        Debug.Log($"[NNW] WebGL assets copied to {destDir}");
    }

    static string FindPluginsWebGLDir()
    {
        var guids = AssetDatabase.FindAssets($"t:Script {nameof(NNWBuildPostProcessor)}");
        foreach (var guid in guids)
        {
            var scriptPath = AssetDatabase.GUIDToAssetPath(guid);
            var editorDir  = Path.GetDirectoryName(scriptPath);
            var packageRoot = Path.GetDirectoryName(editorDir);
            var candidate  = Path.Combine(packageRoot, "Runtime", "Plugins", "WebGL");

            if (Directory.Exists(candidate)) return candidate;
        }
        return null;
    }

    static void CopyIfExists(string sourceDir, string destDir, string filename)
    {
        var src  = Path.Combine(sourceDir, filename);
        var dest = Path.Combine(destDir, filename);
        if (File.Exists(src))
        {
            File.Copy(src, dest, overwrite: true);
            Debug.Log($"[NNW] Copied {filename}");
        }
        else Debug.LogWarning($"[NNW] {filename} not found at {src}, skipping.");
    }
}