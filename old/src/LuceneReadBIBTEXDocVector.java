

import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.File;

/**
 * CAMBIOS
 * - Generar una query con n-terminos que el usuario entrega como parametro
 * - Hacerle el stemming correspondiente a los términos de la consulta
 * - Buscar en cada documento los términos de búsqueda y escribirlos como las primeras filas del documento generado.
 *   Si el termino no esta en el documento, simplemente se escribe el string sin posiciones
 * - Los nombres de archivos deben tener el siguiente formato IIx.txt, donde x es el ranking del documento.
 * - El resto de los términos se agregan a las lista con sus respectivas posiciones. Sin considerar los terminos de busqueda
 *   ya ingresados al principio. 
 * */

/**
 * This is an example for accessing a stored document vector from a Lucene
 * index.
 * 
 * @author Patricio Galeas
 * @version 2017-10-09
 */
public class LuceneReadBIBTEXDocVector {

	private String searchField;
	private String[] queryStrings;

	public LuceneReadBIBTEXDocVector(String field) {
		this.searchField = field;
	}
	
	public LuceneReadBIBTEXDocVector(String[] field_and_queryStrings) {
		this.searchField = field_and_queryStrings[0];
		int j=0;
		queryStrings = new String[field_and_queryStrings.length-1];
		for(int i=1; i<field_and_queryStrings.length; i++) {
			this.queryStrings[j++] = field_and_queryStrings[i];
		}
		
	}
	
	public void setQueryTerms(String[] queryStrings) {
		this.queryStrings = queryStrings;
	}

	public void printTermPositions() {
		try {

			String pathIndex = "index/";

			Directory dir = FSDirectory.open(new File(pathIndex).toPath());
            System.out.println(dir.toString());
			IndexReader index = DirectoryReader.open(dir);

			int totalDocs = index.getDocCount(this.searchField);

			for (int docid = 0; docid < totalDocs; docid++) {

				Document doc = index.document(docid);

				System.out.println("\nDOCUMENT : " + doc.get("docno") + " - " 	+ doc.get("title"));

				// Read the document's document vector
				Terms vector = index.getTermVector(docid, this.searchField); 

				if (vector != null) {

					// You need to use TermsEnum to iterate each entry of the
					// document vector (in alphabetical order).
					
					//System.out.printf("%-25s%-20s\n", "TERM", "POSITIONS");

					TermsEnum terms = vector.iterator();
					PostingsEnum positions = null;
					BytesRef term;
		
					// Create the GDocument for the new index
					GDocument gDoc = new GDocument(docid);
					
					while ((term = terms.next()) != null) {

						String termstr = term.utf8ToString(); // Get the text
																// string
																// of the term.
						long freq = terms.totalTermFreq(); // Get the frequency
															// of
															// the term in the
															// document.

						
						//System.out.printf( "%-20s%-10d", termstr, freq );
						// System.out.printf("%-25s", termstr);
						
						// Add the term (string) to the GDocument
						gDoc.addTerm(termstr);
						


						// Lucene's document vector can also provide the
						// position of the terms
						// (in case you stored these information in the index).
						// Here you are getting a PostingsEnum that includes
						// only one document entry, i.e., the current document.
						positions = terms.postings(positions, PostingsEnum.POSITIONS);
						positions.nextDoc(); // you still need to move the
												// cursor
						// now accessing the occurrence position of the terms by
						// iteratively calling nextPosition()
						for (int i = 0; i < freq; i++) {
							int p = positions.nextPosition();
							//System.out.print((i > 0 ? " " : "") + p);
							// Add the positión of termstr to the gDoc (top of the stack)
							gDoc.addPositions(p);
						}
						
						//System.out.println();

					}
					
					/** Create the text document */
					gDoc.saveDocument(docid+".txt", this.queryStrings);
					
					/** Testing  !!!!!!!
					 * */
					System.out.println("-------"+docid+"------");
					gDoc.saveDocumentFiltered(this.queryStrings);

				} else {
					System.out.println("Positions vector is null !!!");
				}
			}

			index.close();
			dir.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public static void main(String[] args) {

		if (args[0] == null || args.length < 2) {
			System.out
					.println("Error: You need to specify the following parameters"
							+ "\n args[0] the  attribute (title, abstract, keywords, etc.) for displaying the term positions"
							+ "\n args[1] args[2] ... : the query terms");

		} else {
			LuceneReadBIBTEXDocVector vectorPositions = new LuceneReadBIBTEXDocVector(
					args);
			vectorPositions.printTermPositions();
		}

	}

}
