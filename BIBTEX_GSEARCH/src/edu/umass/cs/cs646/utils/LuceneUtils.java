package edu.umass.cs.cs646.utils;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.util.BytesRef;
import org.lemurproject.galago.core.parse.stem.Porter2Stemmer;
import org.lemurproject.galago.krovetz.KStem;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * Some Lucene utilities for CS646 students.
 */
public class LuceneUtils {
	
	public static void main( String[] args ) {
		try {
			
			for ( String word : "Relevance based language models".split( "\\s+" ) ) {
				word = word.toLowerCase();
				//System.out.println( word + "\t" + stemPorter2( word ) + "\t" + stemKrovetz( word ) );
				System.out.println( word + "\t" + "\t" + stemKrovetz( word ) );
			}
			
		} catch ( Exception e ) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Find a document in the index by its docno (external ID).
	 * Returns the internal ID of the document; or -1 if not found.
	 *
	 * @param index      An index reader.
	 * @param fieldDocno The name of the field you used for storing docnos (external document IDs).
	 * @param docno      The docno (external ID) you are looking for.
	 * @return The internal ID of the document in the index; or -1 if not found.
	 * @throws IOException
	 */
	public static int findByDocno( IndexReader index, String fieldDocno, String docno ) throws IOException {
		BytesRef term = new BytesRef( docno );
		PostingsEnum posting = MultiFields.getTermDocsEnum( index, fieldDocno, term, PostingsEnum.NONE );
		if ( posting != null ) {
			int docid = posting.nextDoc();
			if ( docid != PostingsEnum.NO_MORE_DOCS ) {
				return docid;
			}
		}
		return -1;
	}
	
	/**
	 * @param index      An index reader.
	 * @param fieldDocno The name of the field you used for storing docnos (external document IDs).
	 * @param docid      The internal ID of the document
	 * @return The docno (external ID) of the document.
	 * @throws IOException
	 */
	public static String getDocno( IndexReader index, String fieldDocno, int docid ) throws IOException {
		// This implementation is just for you to quickly understand how this works.
		// You should consider reuse the fieldset if you need to read docnos for a lot of documents.
		Set<String> fieldset = new HashSet<>();
		fieldset.add( fieldDocno );
		Document d = index.document( docid, fieldset );
		return d.get( fieldDocno );
	}
	
	//private static Porter2Stemmer porter2 = new Porter2Stemmer();
	private static KStem krovetz = new KStem();
	/*
	public static String stemPorter2( String word ) {
		return porter2.stem( word );
	}
	*/
	public static String stemKrovetz( String word ) {
		return krovetz.stemTerm( word );
	}
	
}
