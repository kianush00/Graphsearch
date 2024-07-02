//package ris.reader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Reader;
import java.io.StringReader;
import java.util.Collection;
import java.util.Map;
import java.util.Scanner;

import org.jbibtex.BibTeXDatabase;
import org.jbibtex.BibTeXParser;
import org.jbibtex.Key;
import org.jbibtex.ParseException;
import org.jbibtex.TokenMgrException;


public class TestBibReader {

	private static final String FILENAME = "tmp/science.bib";
	private static Scanner scanner;

	public static void main(String[] args) {

		String content;
		try {
			
			BibTeXParser parser = new BibTeXParser();
			scanner = new Scanner(new File(FILENAME));
			content = scanner.useDelimiter("\\Z").next();
			Reader reader = new StringReader(content);
			
			BibTeXDatabase database = parser.parseFully(reader);
			
			Map<org.jbibtex.Key, org.jbibtex.BibTeXEntry> entryMap = database.getEntries();

			Collection<org.jbibtex.BibTeXEntry> entries = entryMap.values();
			for(org.jbibtex.BibTeXEntry entry : entries){
				org.jbibtex.Value title = entry.getField(org.jbibtex.BibTeXEntry.KEY_TITLE);
				org.jbibtex.Value author = entry.getField(org.jbibtex.BibTeXEntry.KEY_AUTHOR);
				org.jbibtex.Value ab = entry.getField(new Key("abstract"));

				if(title == null){
					continue;
				}
				
				System.out.println("author:"+ author.toUserString() + " - abstract: "+ab.toUserString());
				// Do something with the title value
			}
			
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (TokenMgrException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
