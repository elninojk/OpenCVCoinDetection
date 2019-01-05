
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class TemplateMatchAll {

	public static void main(String[] args) {
		
		//Loading open cv library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
		//input , templte and output files
		String inFile ="/Users/jerilkuruvila/eclipse-workspace/OpenCVTemplateMatching/super_mario.jpeg";
		String templateFile="/Users/jerilkuruvila/eclipse-workspace/OpenCVTemplateMatching/SuperMarioCoin.jpg";
		String outFile="/Users/jerilkuruvila/eclipse-workspace/OpenCVTemplateMatching/SuperMatch_All.jpg";;
		
		//Matching method
		int match_method =Imgproc.TM_CCOEFF_NORMED;
		
		// read inputs into the mat
		Mat img = Imgcodecs.imread(inFile);
		Mat templ = Imgcodecs.imread(templateFile);
		
		//prepare skelton for resultant image
		int result_cols = img.cols() - templ.cols() + 1;
	    int result_rows = img.rows() - templ.rows() + 1;
	    Mat result = new Mat(result_rows, result_cols, CvType.CV_32FC1);

	    // do the matching
	    Imgproc.matchTemplate(img, templ, result, match_method );
	    
	    // use threshold to restrict the number of results, mine is very poor just .3
	    Imgproc.threshold(result, result,0.3,1,Imgproc.THRESH_TOZERO); 
	    Point matchLoc;
	    Point maxLoc;
	    Point minLoc;

	    MinMaxLocResult mmr;
	    //Iterate through all results
	    while(true)
	    {
	        mmr = Core.minMaxLoc(result);
	        matchLoc = mmr.maxLoc;
	        if(mmr.maxVal >=0.5)
	        {
	        	//Drawing the rectangle in the resultant image
	        	Imgproc.rectangle(img, matchLoc, 
	                new Point(matchLoc.x + templ.cols(),matchLoc.y + templ.rows()), 
	                new    Scalar(0,255,0));
	        	//Removing the already drawn result using that -1
	        	Imgproc.rectangle(result, matchLoc, 
	                new Point(matchLoc.x + templ.cols(),matchLoc.y + templ.rows()), 
	                new    Scalar(0,255,0),-1);
	            //break;
	        }
	        else
	        {
	            break; //No more results within tolerance, break search
	        }
	    }
	    
		// Save the visualized detection.
	    System.out.println("Writing " + outFile);
	    Imgcodecs.imwrite(outFile, img);


	}

}
