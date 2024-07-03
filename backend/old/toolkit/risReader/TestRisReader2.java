//package ris.reader;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import de.undercouch.citeproc.ris.RISLibrary;
import de.undercouch.citeproc.ris.RISParser;
import de.undercouch.citeproc.ris.RISReference;

public class TestRisReader2 {

	private static final String FILENAME = "tmp/example1.ris";
	//private static Scanner scanner;

	public static void main(String[] args) {

		BufferedReader br = null;
		try {
			
			br = new BufferedReader(new FileReader(FILENAME));

			String sCurrentLine;
			String publicationsString = "";
			
			while ((sCurrentLine = br.readLine()) != null) {
			
				String pattern = "TY|A1|A2|A3|A4|AB|AD|AN|AU|AV|BT|C1|C2|C3|C4|C5|C6|C7|C8|CA|CN|CP|CT|CY|DA|DB|DO|DP|ED|EP|ET|ID|IS|J1|J2|JA|JF|JO|KW|L1|L2|L3|L4|LA|LB|LK|M1|M2|M3|N1|N2|NV|OP|PB|PP|PY|RI|RN|RP|SE|SN|SP|ST|T1|T2|T3|TA|TI|TT|U1|U2|U3|U4|U5|UR|VL|VO|Y1|Y2|ER  - (.+?)";
				
				// Create the patterns
				Pattern symbolsPattern = Pattern.compile(pattern);

				// Now create matcher objects.
				Matcher startMatch = symbolsPattern.matcher(sCurrentLine);
	
				if (startMatch.find()) { // first attribute
					//System.out.println("["+sCurrentLine.length()+"]"+sCurrentLine + "\n");
					if(sCurrentLine.length()>6)
						publicationsString += sCurrentLine + "\n"; 
				}
			}
			
System.out.println(publicationsString);
			
			Reader r = new StringReader(publicationsString);

			RISParser parser = new RISParser();
			
			RISLibrary l = parser.parse(r);

			List<RISReference> references = l.getReferences();

			Iterator<RISReference> itr = references.iterator();

			while (itr.hasNext()) {
				RISReference ref = itr.next();
				System.out.println("title:"+ref.getPrimaryTitle());
			}
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			//System.out.println("error");
		}



	}

}
