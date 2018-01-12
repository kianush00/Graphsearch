package edu.umass.cs.cs646.tutorial.lucene;

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
public class LuceneReadBIBTEXDocVector_OLD {

	private String searchField;

	public LuceneReadBIBTEXDocVector_OLD(String field) {
		this.searchField = field;
	}

	public void printTermPositions() {

		try {

			String pathIndex = "index";

			Directory dir = FSDirectory.open(new File(pathIndex).toPath());
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
					
					System.out.printf("%-25s%-20s\n", "TERM", "POSITIONS");

					TermsEnum terms = vector.iterator();
					PostingsEnum positions = null;
					BytesRef term;
					while ((term = terms.next()) != null) {

						String termstr = term.utf8ToString(); // Get the text
																// string
																// of the term.
						long freq = terms.totalTermFreq(); // Get the frequency
															// of
															// the term in the
															// document.
						// System.out.printf( "%-20s%-10d", termstr, freq );
						System.out.printf("%-25s", termstr);

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
							System.out.print((i > 0 ? " " : "")
									+ positions.nextPosition());
						}
						System.out.println();

					}

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

		if (args[0] == null) {
			System.out
					.println("Error: You need to specify as parameter args[0] the  attribute (title, abstract, keywords, etc.) for displaying the term positions...");
		} else {

			LuceneReadBIBTEXDocVector_OLD vectorPositions = new LuceneReadBIBTEXDocVector_OLD(
					args[0]);
			vectorPositions.printTermPositions();
		}

	}

}
