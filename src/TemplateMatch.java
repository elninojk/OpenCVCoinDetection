
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class TemplateMatch {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
		String inFile ="/Users/jerilkuruvila/eclipse-workspace/OpenCVTemplateMatching/super_mario.jpeg";
		String templateFile="/Users/jerilkuruvila/eclipse-workspace/OpenCVTemplateMatching/SuperMarioCoin.jpg";
		String outFile="/Users/jerilkuruvila/eclipse-workspace/OpenCVTemplateMatching/SuperMatch.jpg";;
		
		int match_method =Imgproc.TM_CCOEFF;
		Mat img = Imgcodecs.imread(inFile);
		Mat templ = Imgcodecs.imread(templateFile);
		
		int result_cols = img.cols() - templ.cols() + 1;
	    int result_rows = img.rows() - templ.rows() + 1;
	    Mat result = new Mat(result_rows, result_cols, CvType.CV_32FC1);

	    
		// / Do the Matching and Normalize
	    Imgproc.matchTemplate(img, templ, result, match_method );
	    Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());

	    // / Localizing the best match with minMaxLoc
	    MinMaxLocResult mmr = Core.minMaxLoc(result);

	    Point matchLoc;
	    if (match_method == Imgproc.TM_SQDIFF
	            || match_method == Imgproc.TM_SQDIFF_NORMED) {
	        matchLoc = mmr.minLoc;
	    } else {
	        matchLoc = mmr.maxLoc;
	    }

	    // / Show me what you got
	    Imgproc.rectangle(img, matchLoc, new Point(matchLoc.x + templ.cols(),
	            matchLoc.y + templ.rows()), new Scalar(0, 255, 0));

	    
		// Save the visualized detection.
	    System.out.println("Writing " + outFile);
	    Imgcodecs.imwrite(outFile, img);


	}

}
