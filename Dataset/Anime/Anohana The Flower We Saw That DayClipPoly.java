import java.awt.*;
import java.awt.event.*;
import java.util.*;

public class ClipPoly extends Frame
{  public static void main(String[] args){new ClipPoly();}

   ClipPoly()
   {  super("Define polygon vertices by clicking");
      addWindowListener(new WindowAdapter()
         {public void windowClosing(WindowEvent e){System.exit(0);}});
      setSize(500, 300);
      add("Center", new CvClipPoly());
      setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
      show();
   }
}

class CvClipPoly extends Canvas
{  Poly poly = null;
   float rWidth = 10.0F, rHeight = 7.5F, pixelSize;
   int X0, Y0, centerX, centerY;
   boolean ready = true;

   CvClipPoly()
   {  addMouseListener(new MouseAdapter()
      {  public void mousePressed(MouseEvent evt)
         {  int X = evt.getX(), Y = evt.getY();
            if (ready)
            {  poly = new Poly();
               X0 = X; Y0 = Y;
               ready = false;
            }
            float x = fx(X), y = fy(Y);
            if (poly.size() > 0 &&
               Math.abs(X - X0) < 3 && Math.abs(Y - Y0) < 3)
               ready = true;
            else
               poly.addVertex(new Point2D(x, y));
            repaint();
         }
      });
   }

   void initgr()
   {  Dimension d = getSize();
      int maxX = d.width - 1, maxY = d.height - 1;
      pixelSize = Math.max(rWidth/maxX, rHeight/maxY);
      centerX = maxX/2; centerY = maxY/2;
   }

   int iX(float x){return Math.round(centerX + x/pixelSize);}
   int iY(float y){return Math.round(centerY - y/pixelSize);}
   float fx(int X){return (X - centerX) * pixelSize;}
   float fy(int Y){return (centerY - Y) * pixelSize;}

   void drawLine(Graphics g, float xP, float yP, float xQ, float yQ)
   {  g.drawLine(iX(xP), iY(yP), iX(xQ), iY(yQ));
   }

   void drawPoly(Graphics g, Poly poly)
   {  int n = poly.size();
      if (n == 0) return;
      Point2D A = poly.vertexAt(n - 1);
      for (int i=0; i<n; i++)
      {  Point2D B = poly.vertexAt(i);
         drawLine(g, A.x, A.y, B.x, B.y);
         A = B;
      }
   }

   public void paint(Graphics g)
   {  initgr();
      float xmin = -rWidth/3, xmax = rWidth/3,
            ymin = -rHeight/3, ymax = rHeight/3;
      // Draw clipping rectangle:
      g.setColor(Color.blue);
      drawLine(g, xmin, ymin, xmax, ymin);
      drawLine(g, xmax, ymin, xmax, ymax);
      drawLine(g, xmax, ymax, xmin, ymax);
      drawLine(g, xmin, ymax, xmin, ymin);
      g.setColor(Color.black);
      if (poly == null) return;
      int n = poly.size();
      if (n == 0) return;
      Point2D A = poly.vertexAt(0);

      if (!ready)
      {  // Show tiny rectangle around first vertex:
         g.drawRect(iX(A.x)-2, iY(A.y)-2, 4, 4);
         // Draw incomplete polygon:
         for (int i=1; i<n; i++)
         {  Point2D B = poly.vertexAt(i);
            drawLine(g, A.x, A.y, B.x, B.y);
            A = B;
         }
      }
      else
      {  poly.clip(xmin, ymin, xmax, ymax);
         drawPoly(g, poly);
      }
   }
}

class Poly
{  Vector v = new Vector();
   void addVertex(Point2D P){v.addElement(P);}
   int size(){return v.size();}

   Point2D vertexAt(int i)
   {  return (Point2D)v.elementAt(i);
   }

   void clip(float xmin, float ymin, float xmax, float ymax)
   {  // Sutherland-Hodgman polygon clipping:
      Poly poly1 = new Poly();
      int n;
      Point2D A, B;
      boolean sIns, pIns;

      // Clip against x == xmax:
      if ((n = size()) == 0) return;
      B = vertexAt(n-1);
      for (int i=0; i<n; i++)
      {  A = B; B = vertexAt(i);
         sIns = A.x <= xmax; pIns = B.x <= xmax;
         if (sIns != pIns)
            poly1.addVertex(new Point2D(xmax, A.y +
            (B.y - A.y) * (xmax - A.x)/(B.x - A.x)));
         if (pIns) poly1.addVertex(B);
      }
      v = poly1.v; poly1 = new Poly();

      // Clip against x == xmin:
      if ((n = size()) == 0) return;
      B = vertexAt(n-1);
      for (int i=0; i<n; i++)
      {  A = B; B = vertexAt(i);
         sIns = A.x >= xmin; pIns = B.x >= xmin;
         if (sIns != pIns)
            poly1.addVertex(new Point2D(xmin, A.y +
            (B.y - A.y) * (xmin - A.x)/(B.x - A.x)));
         if (pIns) poly1.addVertex(B);
      }
      v = poly1.v; poly1 = new Poly();

      // Clip against y == ymax:
      if ((n = size()) == 0) return;
      B = vertexAt(n-1);
      for (int i=0; i<n; i++)
      {  A = B; B = vertexAt(i);
         sIns = A.y <= ymax; pIns = B.y <= ymax;
         if (sIns != pIns)
            poly1.addVertex(new Point2D(A.x +
            (B.x - A.x) * (ymax - A.y)/(B.y - A.y), ymax));
         if (pIns) poly1.addVertex(B);
      }
      v = poly1.v; poly1 = new Poly();

      // Clip against y == ymin:
      if ((n = size()) == 0) return;
      B = vertexAt(n-1);
      for (int i=0; i<n; i++)
      {  A = B; B = vertexAt(i);
         sIns = A.y >= ymin; pIns = B.y >= ymin;
         if (sIns != pIns)
            poly1.addVertex(new Point2D(A.x +
            (B.x - A.x) * (ymin - A.y)/(B.y - A.y), ymin));
         if (pIns) poly1.addVertex(B);
      }
      v = poly1.v; poly1 = new Poly();
   }
}
