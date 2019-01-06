
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class VideoCaptured {

	public static void main(String[] args) throws Exception {
		String xmlFile = "/Users/jerilkuruvila/eclipse-workspace/OpenCVTemplateMatching/cascade.xml";
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		CascadeClassifier faceDetector = new CascadeClassifier(xmlFile); 
		Mat frame = new Mat();
	    //0; default video device id
	    VideoCapture camera = new VideoCapture(0);
	    JFrame jframe = new JFrame("Title");
	    jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	    JLabel vidpanel = new JLabel();
	    jframe.setContentPane(vidpanel);
	    jframe.setVisible(true);

	    
	    
	    while (true) {
	        if (camera.read(frame)) {
	            ImageIcon image = new ImageIcon(Mat2BufferedImage(frame,faceDetector));
	            vidpanel.setIcon(image);
	            vidpanel.repaint();

	        }
	    }
	}
	
	public static BufferedImage Mat2BufferedImage(Mat m,CascadeClassifier faceDetector ) throws Exception {
	    //Method converts a Mat to a Buffered Image
	    int type = BufferedImage.TYPE_BYTE_GRAY;
	     if ( m.channels() > 1 ) {
	         type = BufferedImage.TYPE_3BYTE_BGR;
	     }
	     int bufferSize = m.channels()*m.cols()*m.rows();
	     byte [] b = new byte[bufferSize];
	     m.get(0,0,b); // get all the pixels
	     BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
	     final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
	     System.arraycopy(b, 0, targetPixels, 0, b.length);  
	     
	   
	     MatOfRect faceDetections = new MatOfRect(); 
	     Mat mm = bufferedImageToMat(image);
         faceDetector.detectMultiScale(mm, faceDetections); 
         System.out.println("Detections ****"+faceDetections.toArray());
         
      // Creating a rectangular box showing faces detected 
         for (Rect rect : faceDetections.toArray()) 
         { 
             Imgproc.rectangle(mm, new Point(rect.x, rect.y), 
              new Point(rect.x + rect.width, rect.y + rect.height), 
                                            new Scalar(0, 255, 0)); 
         } 
	     return Mat2BufferedImage(mm);
	    }
	
	public static Mat bufferedImageToMat(BufferedImage bi) {
		  Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		  byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		  mat.put(0, 0, data);
		  return mat;
		}
	
	static BufferedImage Mat2BufferedImage(Mat matrix)throws Exception {        
	    MatOfByte mob=new MatOfByte();
	    Imgcodecs.imencode(".jpg", matrix, mob);
	    byte ba[]=mob.toArray();

	    BufferedImage bi=ImageIO.read(new ByteArrayInputStream(ba));
	    return bi;
	}

}
