//package risReader;

import static org.junit.Assert.assertEquals;

import java.io.Reader;
import java.io.StringReader;

import org.junit.Test;

import de.undercouch.citeproc.ris.RISLibrary;
import de.undercouch.citeproc.ris.RISParser;
import de.undercouch.citeproc.ris.RISReference;
import de.undercouch.citeproc.ris.RISType;

/**
 * Tests {@link RISParser}
 * @author Michel Kraemer
 */
public class RISParserTest {
	/**
	 * Tests if a single entry can be parsed correctly
	 * @throws Exception if anything goes wrong
	 */
	@Test
	public void singleEntry() throws Exception {
		String e =
			"TY  - RPRT\n" + 
			"AU  - Johnson, S. C.\n" + 
			"AU  - Kernighan, B. W.\n" + 
			"PY  - 1973//\n" + 
			"TI  - The Programming Language B\n" + 
			"IS  - 8\n" + 
			"PB  - Bell Laboratories,\n" + 
			"CY  - Murray Hill, NJ, USA\n" + 
			"ID  - Johnson:1973:PLB\n" + 
			"ER  - ";
		
		
		Reader r = new StringReader(e);
		RISParser parser = new RISParser();
		RISLibrary l = parser.parse(r);
		assertEquals(1, l.getReferences().size());
		
		RISReference ref = l.getReferences().get(0);
		
		assertEquals(RISType.RPRT, ref.getType());
		assertEquals("The Programming Language B", ref.getTitle());
		assertEquals("1973//", ref.getYear());
		assertEquals("8", ref.getIssue());
		assertEquals("Bell Laboratories,", ref.getPublisher());
		assertEquals("Murray Hill, NJ, USA", ref.getPlace());
		assertEquals("Johnson:1973:PLB", ref.getId());
		assertEquals(2, ref.getAuthors().length);
		assertEquals("Johnson, S. C.", ref.getAuthors()[0]);
		assertEquals("Kernighan, B. W.", ref.getAuthors()[1]);
	}
}
