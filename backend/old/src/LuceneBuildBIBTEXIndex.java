

import java.io.File;
import java.io.Reader;
import java.io.StringReader;
import java.util.Collection;
import java.util.Map;
import java.util.Scanner;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.en.KStemFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.jbibtex.BibTeXDatabase;
import org.jbibtex.BibTeXParser;
import org.jbibtex.Key;

/**
 * This is an example for building a Lucene index from a BibTex file.
 *
 * @author Patricio Galeas
 * @version 2017-10-09
 */
public class LuceneBuildBIBTEXIndex {
	
	private String FILENAME;
	private Scanner scanner;
	
	public LuceneBuildBIBTEXIndex(String filename) {
		if (filename != null) {
			FILENAME = filename;
		}
	}
	
	public void initialize() {
		this.createIndexDirectory();
	}
	
	private void createIndexDirectory() {
		
		File theDir = new File("index");

		// if the directory does not exist, create it
		if (!theDir.exists()) {
		    System.out.println("creating directory: " + theDir.getName());
		    boolean result = false;

		    try{
		        theDir.mkdir();
		        result = true;
		    } 
		    catch(SecurityException se){
		        //handle it
		    }        
		    if(result) System.out.println("Directory  \"index\"  was created");  
		    else 	System.out.println("Directory \"index\" already exists. ");
		}
	}
	
	public void buildIndex() {
		try {
			
			String pathIndex = "index";
			String content;
			
			Directory dir = FSDirectory.open( new File( pathIndex ).toPath());
			
			// Analyzer includes options for text processing
			Analyzer analyzer = new Analyzer() {
				protected TokenStreamComponents createComponents( String fieldName ) {
					// Step 1: tokenization (Lucene's StandardTokenizer is suitable for most text retrieval occasions)
					TokenStreamComponents ts = new TokenStreamComponents( new StandardTokenizer() );
					// Step 2: transforming all tokens into lowercased ones
					ts = new TokenStreamComponents( ts.getTokenizer(), new LowerCaseFilter( ts.getTokenStream() ) );
					// Step 3: whether to remove stop words
					// Uncomment the following line to remove stop words
					ts = new TokenStreamComponents( ts.getTokenizer(), new StopFilter( ts.getTokenStream(), StandardAnalyzer.ENGLISH_STOP_WORDS_SET ) );
					// Step 4: whether to apply stemming
					// Uncomment the following line to apply Krovetz or Porter stemmer
					ts = new TokenStreamComponents( ts.getTokenizer(), new KStemFilter( ts.getTokenStream() ) );
					//ts = new TokenStreamComponents( ts.getTokenizer(), new PorterStemFilter( ts.getTokenStream() ) );
					//ts = new TokenStreamComponents( ts.getTokenizer(), new SnowballFilter( ts.getTokenStream(), "English" ) );
					return ts;
				}
			};
			
			IndexWriterConfig config = new IndexWriterConfig( analyzer );
			
			// Note that IndexWriterConfig.OpenMode.CREATE will override the original index in the folder
			config.setOpenMode( IndexWriterConfig.OpenMode.CREATE );
			
			IndexWriter ixwriter = new IndexWriter( dir, config );
			
			// This is the field setting for metadata field.
			FieldType fieldTypeMetadata = new FieldType();
			fieldTypeMetadata.setOmitNorms( true );
			fieldTypeMetadata.setIndexOptions( IndexOptions.DOCS );
			fieldTypeMetadata.setStored( true );
			fieldTypeMetadata.setTokenized( false );
			fieldTypeMetadata.freeze();
			
			// This is the field setting for normal text field.
			FieldType fieldTypeText = new FieldType();
			fieldTypeText.setIndexOptions( IndexOptions.DOCS_AND_FREQS_AND_POSITIONS );
			fieldTypeText.setStoreTermVectors( true );
			fieldTypeText.setStoreTermVectorPositions( true );
			fieldTypeText.setTokenized( true );
			fieldTypeText.setStored( true );
			fieldTypeText.freeze();
			
			// Read the bib file
			BibTeXParser parser = new BibTeXParser();
			scanner = new Scanner(new File(FILENAME));
			content = scanner.useDelimiter("\\Z").next();
			Reader reader = new StringReader(content);
			
			/* Parse the bib file entries */
			BibTeXDatabase database = parser.parseFully(reader);
			Map<org.jbibtex.Key, org.jbibtex.BibTeXEntry> entryMap = database.getEntries();
			Collection<org.jbibtex.BibTeXEntry> entries = entryMap.values();
			
			for(org.jbibtex.BibTeXEntry entry : entries){
				/* Read BibTex attributes from each reference */
				Key docKey 					= entry.getKey();
				org.jbibtex.Value title 		= entry.getField(org.jbibtex.BibTeXEntry.KEY_TITLE);
				org.jbibtex.Value author 	= entry.getField(org.jbibtex.BibTeXEntry.KEY_AUTHOR);
				org.jbibtex.Value keywords 	= entry.getField(new Key("keywords"));
				org.jbibtex.Value ab 		= entry.getField(new Key("abstract"));

				if(title == null) continue;
				
				/* Replace null object with empty strings */
				String docKeyString = (docKey != null) ? docKey.toString() : "";
				String authorString = (author != null) ? author.toUserString() : "";
				String keywordsString = (keywords != null) ? keywords.toUserString() : "";
				String abstractString = (ab != null) ? ab.toUserString() : "";
				String titleString = (title != null) ? title.toUserString() : "";
				
				/* Print out the indexing process */
				System.out.println("\nIndexing docno : "+ docKeyString);
				System.out.println("author:"+ authorString);
				System.out.println("title:"+ titleString);
				System.out.println("keywords: "+keywordsString);
				System.out.println("abstract: "+ abstractString);
				
				// Create a Document object
				Document d = new Document();
				
				// Add each field to the document with the appropriate field type options
				d.add( new Field( "docno", docKeyString, fieldTypeMetadata ) );
				d.add( new Field( "title", titleString, fieldTypeText ) );
				d.add( new Field( "author", authorString, fieldTypeText ) );
				d.add( new Field( "abstract", abstractString, fieldTypeText ) );
				
				/* TODO: Only the last keyword is included
				 * The original method must be modified to include all
				 * keywords in a single String */
				d.add( new Field( "keywords", keywordsString, fieldTypeText ) );
				
				// Add the document to index.
				ixwriter.addDocument( d );
				
			}
		
			// Close both the index writer and the directory
			ixwriter.close();
			dir.close();
			
		} catch ( Exception e ) {
			e.printStackTrace();
		}	
	}
	
	public static void main( String[] args ) {
		
		if(args[0] == null) {
			System.out.println("Error: You need to define the path of the bibfile as first parameter ...");
		}
		else {
			LuceneBuildBIBTEXIndex bibindex = new LuceneBuildBIBTEXIndex(args[0]);
			bibindex.initialize();
			bibindex.buildIndex();
		}

	}
	
}
