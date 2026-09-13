package org.noesis.roomwalk;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

/** Grants read access only to the single capture artifact selected for sharing. */
public final class CaptureFileProvider extends ContentProvider {
    public boolean onCreate() { return true; }
    private File resolve(Uri uri) throws FileNotFoundException {
        try {
            List<String> parts = uri.getPathSegments();
            if (parts.size() != 2) throw new IOException("Invalid capture URI");
            File root = new File(getContext().getExternalFilesDir(null), "captures").getCanonicalFile();
            File file = new File(new File(root, parts.get(0)), parts.get(1)).getCanonicalFile();
            if (!file.getPath().startsWith(root.getPath() + File.separator) || !file.isFile()) throw new IOException("Capture unavailable");
            if (!(file.getName().endsWith(".zip") || file.getName().equals("capabilities.json") || file.getName().equals("capture_result.json") || file.getName().equals("imu_capture_manifest.json") || file.getName().equals(PairedCapture.FILE))) throw new IOException("Only exported artifacts may be shared");
            return file;
        } catch (IOException e) { throw new FileNotFoundException(e.getMessage()); }
    }
    public String getType(Uri uri) { return uri.toString().endsWith(".json") ? "application/json" : "application/zip"; }
    public Cursor query(Uri uri, String[] projection, String selection, String[] args, String order) {
        try {
            File f = resolve(uri);
            String[] cols = projection == null ? new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE} : projection;
            MatrixCursor cursor = new MatrixCursor(cols);
            Object[] row = new Object[cols.length];
            for (int i=0;i<cols.length;i++) row[i] = OpenableColumns.DISPLAY_NAME.equals(cols[i]) ? f.getName() : OpenableColumns.SIZE.equals(cols[i]) ? f.length() : null;
            cursor.addRow(row); return cursor;
        } catch (FileNotFoundException e) { return null; }
    }
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        if (!"r".equals(mode)) throw new FileNotFoundException("Read only");
        return ParcelFileDescriptor.open(resolve(uri), ParcelFileDescriptor.MODE_READ_ONLY);
    }
    public Uri insert(Uri uri, ContentValues values) { throw new UnsupportedOperationException(); }
    public int update(Uri uri, ContentValues values, String s, String[] a) { throw new UnsupportedOperationException(); }
    public int delete(Uri uri, String s, String[] a) { throw new UnsupportedOperationException(); }
}
